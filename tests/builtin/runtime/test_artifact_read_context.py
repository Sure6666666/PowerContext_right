# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The shared read context preserves scope, family and caller permissions."""

import asyncio
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from typing import cast

import pytest
from sqlalchemy import insert, select

from powercontext.artifact_plugins import GenerationContext
from powercontext.artifacts import ArtifactRef
from powercontext.artifacts.search import ArtifactSearchUnsupported
from powercontext.builtin.artifacts.profile.models import Profile, ProfileContent, ProfileGeneration
from powercontext.builtin.persistence.artifact_indexes import ArtifactIndexBuilder
from powercontext.builtin.persistence.artifacts import ArtifactRepository, RepositoryArtifactDraft
from powercontext.builtin.persistence.custom_artifacts import (
    StoredArtifactContent,
    custom_artifact_repository,
)
from powercontext.builtin.persistence.sqlite import SQLiteConfig, SQLiteProfile
from powercontext.builtin.persistence.sqlite.experience_index import SQLiteExperienceFTSIndex
from powercontext.builtin.persistence.tables import ARTIFACT_FAMILY_SCHEMAS_TABLE, BUILTIN_TABLES, SOURCE_CURSORS_TABLE
from powercontext.builtin.records import BaseValueNotFoundError, InvalidBaseAccessRequestError
from powercontext.builtin.runtime.artifact_reads import ArtifactReadSession
from powercontext.server.authz import (
    AccessAuditContext,
    AccessControlService,
    AccessDecision,
    AccessDeniedError,
    AccessRole,
    BuiltinAuthorizationProvider,
    CreateBinding,
    PrincipalRef,
    ResourceRef,
)
from powercontext.server.authz.composition import open_builtin_access_control
from powercontext.server.authz.profiles import register_custom_artifact_access_family
from powercontext.server.authz.repository import RelationalAccessRepository


class FilteredProvider(BuiltinAuthorizationProvider):
    async def check_batch(self, requests, /):
        decisions = await super().check_batch(requests)
        return tuple(
            AccessDecision(False, "restricted-artifact", decision.policy_revision)
            if request.resource.artifact_id == "a-hidden"
            else decision
            for request, decision in zip(requests, decisions, strict=True)
        )


def test_generation_reads_share_permission_filtered_candidates_and_exact_evidence(tmp_path):
    async def scenario():
        admin = PrincipalRef(type="service", id="admin")
        alice, bob = PrincipalRef(type="user", id="alice"), PrincipalRef(type="user", id="bob")
        audit = AccessAuditContext(transport="test", operation="artifact-reads")
        family = "context-items"
        register_custom_artifact_access_family(family)
        config = SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'reads.db'}")
        async with AsyncExitStack() as resources:
            profile = await resources.enter_async_context(SQLiteProfile.open(config, tables=BUILTIN_TABLES))
            access = await resources.enter_async_context(
                open_builtin_access_control(config, bootstrap_administrators=(admin,))
            )
            await access.create_binding(
                admin,
                CreateBinding(
                    subject=alice,
                    resource=ResourceRef.scope("scope"),
                    role=AccessRole.SCOPE_VIEWER,
                    idempotency_key="scope-viewer",
                ),
                context=audit,
            )
            repository = custom_artifact_repository(family)
            async with profile.database.transaction() as connection:
                await SQLiteExperienceFTSIndex().initialize(connection)
                await connection.execute(
                    insert(ARTIFACT_FAMILY_SCHEMAS_TABLE).values(
                        family=family,
                        schema_version="1",
                        declaration={"search_fields": ["text"], "schema_version": "1"},
                    )
                )
                for scope, identity in (("scope", "a-hidden"), ("scope", "b-visible"), ("other", "foreign")):
                    await repository.create(
                        connection,
                        scope,
                        identity,
                        RepositoryArtifactDraft(
                            family=family,
                            content=StoredArtifactContent(schema_version="1", content={"text": "数据库 TIMEOUT"}),
                        ),
                    )
            builder = ArtifactIndexBuilder(profile.database)
            await builder.refresh("scope", family)
            assert await builder.build("scope", family)
            for identity, owner in (("a-hidden", bob), ("b-visible", alice)):
                await access.establish_artifact_owner(
                    ResourceRef.artifact("scope", family=family, artifact_id=identity),
                    owner,
                    idempotency_key=identity,
                    context=audit,
                )
            access = AccessControlService(
                FilteredProvider(cast(RelationalAccessRepository, access.relationships)),
                relationships=access.relationships,
                audit=access.audit,
            )
            reads = ArtifactReadSession(
                profile.database, "scope", (family,), access=access, principal=alice, audit=audit
            )
            context = GenerationContext(
                "scope", resources, reads.read, list_artifacts=reads.list, search_artifacts=reads.search
            )
            page = await context.list_artifacts(family, limit=1)
            assert [item.artifact_id for item in page.items] == ["b-visible"]
            assert page.next_cursor is None
            matches = await context.search_artifacts(family, "数据库", limit=1)
            assert [hit.artifact_ref.artifact_id for hit in matches.matches] == ["b-visible"]
            assert matches.artifacts is not None
            assert matches.artifacts[0].content.model_dump() == {"text": "数据库 TIMEOUT"}
            assert not reads.refs
            snapshot = await context.read_revision(matches.matches[0].artifact_ref)
            assert snapshot.content["text"] == "数据库 TIMEOUT"
            assert reads.refs == {(family, "b-visible", 1)}
            with pytest.raises(AccessDeniedError):
                await context.read_latest(family, "a-hidden")
            with pytest.raises(InvalidBaseAccessRequestError):
                await context.read_latest("undeclared", "any")
            with pytest.raises(InvalidBaseAccessRequestError):
                await context.list_artifacts("undeclared")
            with pytest.raises(InvalidBaseAccessRequestError):
                await context.search_artifacts("undeclared", "数据库")
            assert await context.read_latest(family, "foreign") is None
            denied = ArtifactReadSession(
                profile.database, "other", (family,), access=access, principal=alice, audit=audit
            )
            foreign_context = GenerationContext(
                "other", resources, denied.read, list_artifacts=denied.list, search_artifacts=denied.search
            )
            with pytest.raises(AccessDeniedError):
                await foreign_context.read_latest(family, "foreign")
            with pytest.raises(AccessDeniedError):
                await foreign_context.list_artifacts(family)
            with pytest.raises(AccessDeniedError):
                await foreign_context.search_artifacts(family, "数据库")

    asyncio.run(scenario())


def test_generation_context_reads_fixed_history_across_families_and_pages_current_references(tmp_path):
    async def scenario():
        config = SQLiteConfig(url=f"sqlite+aiosqlite:///{tmp_path / 'history.db'}")
        async with AsyncExitStack() as resources:
            profile = await resources.enter_async_context(SQLiteProfile.open(config, tables=BUILTIN_TABLES))
            items = custom_artifact_repository("context-items")
            roots = custom_artifact_repository("context-roots")
            async with profile.database.transaction() as connection:
                await SQLiteExperienceFTSIndex().initialize(connection)
                for family in ("context-items", "context-roots"):
                    await connection.execute(
                        insert(ARTIFACT_FAMILY_SCHEMAS_TABLE).values(
                            family=family,
                            schema_version="1",
                            declaration={
                                "schema_version": "1",
                                "search_fields": ["text"] if family == "context-items" else [],
                            },
                        )
                    )
                first = await items.create(
                    connection,
                    "scope",
                    "a",
                    RepositoryArtifactDraft(
                        family="context-items",
                        content=StoredArtifactContent(schema_version="1", content={"text": "旧证据"}),
                    ),
                )
                await roots.create(
                    connection,
                    "scope",
                    "root",
                    RepositoryArtifactDraft(
                        family="context-roots",
                        content=StoredArtifactContent(
                            schema_version="1", content={"ref": first.as_ref().model_dump(mode="json")}
                        ),
                    ),
                )
                for identity in ("b", "c"):
                    await items.create(
                        connection,
                        "scope",
                        identity,
                        RepositoryArtifactDraft(
                            family="context-items",
                            content=StoredArtifactContent(schema_version="1", content={"text": "其他"}),
                        ),
                    )
                builtin = await ArtifactRepository((Profile,)).create(
                    connection,
                    "scope",
                    "profile",
                    RepositoryArtifactDraft(
                        family="profile",
                        content=ProfileContent(
                            content="使用中文",
                            generation=ProfileGeneration(mode="manual_create", created_at=datetime.now(UTC)),
                        ),
                    ),
                )
                await connection.execute(
                    insert(SOURCE_CURSORS_TABLE).values(
                        scope_id="scope", binding_name="reports", cursor=b'{"through":7}', generation=3
                    )
                )
            reads = ArtifactReadSession(profile.database, "scope", ("context-items", "context-roots", "profile"))
            context = GenerationContext(
                "scope", resources, reads.read, list_artifacts=reads.list, search_artifacts=reads.search
            )
            root = await context.read_latest("context-roots", "root")
            assert root is not None
            fixed_ref = ArtifactRef.model_validate(root.content["ref"])
            async with profile.database.transaction() as connection:
                await items.revise(
                    connection,
                    "scope",
                    first,
                    RepositoryArtifactDraft(
                        family="context-items",
                        content=StoredArtifactContent(schema_version="1", content={"text": "数据库 NEW"}),
                    ),
                )
            builder = ArtifactIndexBuilder(profile.database)
            await builder.refresh("scope", "context-items")
            assert await builder.build("scope", "context-items")
            first_page = await context.list_artifacts("context-items", limit=2)
            assert [(ref.artifact_id, ref.revision) for ref in first_page.items] == [("a", 2), ("b", 1)]
            assert first_page.next_cursor is not None
            last_page = await context.list_artifacts("context-items", limit=2, cursor=first_page.next_cursor)
            assert [(ref.artifact_id, ref.revision) for ref in last_page.items] == [("c", 1)]
            assert last_page.next_cursor is None
            assert not (await context.search_artifacts("context-items", "旧证据")).matches
            current = await context.search_artifacts("context-items", "数据库")
            assert current.matches[0].artifact_ref.revision == 2
            assert (await context.read_revision(current.matches[0].artifact_ref)).content["text"] == "数据库 NEW"
            assert (await context.read_revision(fixed_ref)).content["text"] == "旧证据"
            snapshot = await context.read_revision(builtin.as_ref())
            assert snapshot.schema_version is None
            assert snapshot.content == builtin.content.model_dump(mode="json", by_alias=True)
            with pytest.raises(BaseValueNotFoundError):
                await context.read_revision(ArtifactRef(family="context-items", artifact_id="a", revision=99))
            with pytest.raises(ArtifactSearchUnsupported):
                await context.search_artifacts("context-items", "数据库", mode="hybrid")
            async with profile.database.transaction() as connection:
                cursor = (await connection.execute(select(SOURCE_CURSORS_TABLE))).mappings().one()
                assert cursor["cursor"] == b'{"through":7}' and cursor["generation"] == 3

    asyncio.run(scenario())
