"""
Composio SDK tools.
"""

import base64
import hashlib
import itertools
import json
import os
import time
import typing as t
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import plotly.express as px
import pandas as pd

from pydantic import BaseModel

from composio import Action, ActionType, App, AppType, TagType
from composio.client import Composio
from composio.client.collections import (
    ActionModel,
    ConnectedAccountModel,
    FileModel,
    SuccessExecuteActionResponseModel,
    TriggerSubscription,
)
from composio.client.exceptions import ComposioClientError
from composio.constants import (
    DEFAULT_ENTITY_ID,
    ENV_COMPOSIO_API_KEY,
    LOCAL_CACHE_DIRECTORY,
    LOCAL_OUTPUT_FILE_DIRECTORY_NAME,
    USER_DATA_FILE_NAME,
)
from composio.exceptions import ApiKeyNotProvidedError, ComposioSDKError
from composio.storage.user import UserData
from composio.tools.env.factory import ExecEnv, WorkspaceFactory
from composio.tools.local.base import Action as LocalAction
from composio.tools.local.handler import LocalClient
from composio.utils.enums import get_enum_key
from composio.utils.logging import WithLogger
from composio.utils.url import get_api_url_base
from composio.utils.logging import get as get_logger


output_dir = LOCAL_CACHE_DIRECTORY / LOCAL_OUTPUT_FILE_DIRECTORY_NAME
logger = get_logger("ComposioToolSet")

class ComposioToolSet(WithLogger):
    """Composio toolset."""

    _remote_client: t.Optional[Composio] = None
    _connected_accounts: t.Optional[t.List[ConnectedAccountModel]] = None
    _run_start_timestamp: t.Optional[float] = None
    _logging_enabled: t.Optional[bool] = False
    _server_active: t.Optional[bool] = False

    def __init__(
        self,
        api_key: t.Optional[str] = None,
        base_url: t.Optional[str] = None,
        runtime: t.Optional[str] = None,
        output_in_file: bool = False,
        entity_id: str = DEFAULT_ENTITY_ID,
        workspace_env: ExecEnv = ExecEnv.HOST,
        workspace_id: t.Optional[str] = None,
        action_logging: t.Optional[bool] = True,
    ) -> None:
        """
        Initialize composio toolset

        :param api_key: Composio API key
        :param base_url: Base URL for the Composio API server
        :param runtime: Name of the framework runtime, eg. openai, crewai...
        :param output_in_file: Whether to output the result to a file.
        :param entity_id: The ID of the entity to execute the action on.
            Defaults to "default".
        :param workspace_env: Environment where actions should be executed,
            you can choose from `host`, `docker`, `flyio` and `e2b`.
        :param workspace_id: Workspace ID for loading an existing workspace
        :param action_logging: Whether to log the actions or not
        """
        super().__init__()
        self.entity_id = entity_id
        self.output_in_file = output_in_file
        self.base_url = base_url

        try:
            self.api_key = (
                api_key
                or os.environ.get(ENV_COMPOSIO_API_KEY)
                or UserData.load(LOCAL_CACHE_DIRECTORY / USER_DATA_FILE_NAME).api_key
            )
        except FileNotFoundError:
            self.logger.debug("`api_key` is not set when initializing toolset.")

        if workspace_id is None:
            self.logger.debug(
                f"Workspace ID not provided, using `{workspace_env}` "
                "to create a new workspace"
            )
            self.workspace = WorkspaceFactory.new(
                wtype=workspace_env,
                composio_api_key=self.api_key,
                composio_base_url=base_url or get_api_url_base(),
            )
        else:
            self.logger.debug(f"Loading workspace with ID: {workspace_id}")
            self.workspace = WorkspaceFactory.get(
                id=workspace_id,
            )

        self._runtime = runtime
        self._local_client = LocalClient()

        if action_logging and not ComposioToolSet._logging_enabled:
            try:
                import plotly.express as px
                import pandas as pd
            except ImportError:
                self.logger.error(
                    "Plotly and pandas are required for action logging. "
                    "Please install them using `pip install plotly pandas`"
                )
                return
            self.start_logging_server()

    def start_logging_server(self):
        if not ComposioToolSet._server_active:
            server_thread = threading.Thread(target=self._run_server)
            server_thread.start()

    def _run_server(self):
        self.logger.info("Action logging is enabled, visit http://localhost:8032 to view the logs")
        server_address = ('', 8032)
        httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
        ComposioToolSet._server_active = True
        httpd.serve_forever()

    def _load_logs(self, log_dir):
        logs = []
        for log_file in log_dir.glob("*.json"):
            with open(log_file, "r") as file:
                logs.extend(json.load(file))
        return logs

    def _prepare_data(self, logs):
        data = []
        for log in logs:
            data.append({
                "action": log["action"],
                "start_time": log["start_time"],
                "end_time": log["end_time"],
                "duration": log["duration"],
                "status": log["status"],
                "error": log.get("error", None)
            })
        return pd.DataFrame(data)

    def set_workspace_id(self, workspace_id: str) -> None:
        self.workspace = WorkspaceFactory.get(id=workspace_id)

    @property
    def client(self) -> Composio:
        if self.api_key is None:
            raise ApiKeyNotProvidedError()

        if self._remote_client is None:
            self._remote_client = Composio(
                api_key=self.api_key,
                base_url=self.base_url,
                runtime=self.runtime,
            )
            self._remote_client.local = self._local_client

        return self._remote_client

    @property
    def runtime(self) -> t.Optional[str]:
        return self._runtime

    @classmethod
    def reset_run_timestamp(cls) -> None:
        """Reset the run timestamp."""
        cls._run_start_timestamp = None

    def check_connected_account(self, action: ActionType) -> None:
        """Check if connected account is required and if required it exists or not."""
        action = Action(action)
        if action.no_auth:
            return

        if self._connected_accounts is None:
            self._connected_accounts = t.cast(
                t.List[ConnectedAccountModel],
                self.client.connected_accounts.get(),
            )

        if action.app not in [
            connection.appUniqueId for connection in self._connected_accounts
        ]:
            raise ComposioSDKError(
                f"No connected account found for app `{action.app}`; "
                f"Run `composio add {action.app}` to fix this"
            )

    def _execute_local(
        self,
        action: Action,
        params: t.Dict,
        metadata: t.Optional[t.Dict] = None,
    ) -> t.Dict:
        """Execute a local action."""
        response = self.workspace.execute_action(
            action=action,
            request_data=params,
            metadata=metadata or {},
        )
        if isinstance(response, BaseModel):
            return response.model_dump()
        return response

    def _execute_remote(
        self,
        action: Action,
        params: t.Dict,
        entity_id: str = DEFAULT_ENTITY_ID,
        connected_account_id: t.Optional[str] = None,
        text: t.Optional[str] = None,
    ) -> t.Dict:
        """Execute a remote action."""
        self.check_connected_account(
            action=action,
        )
        output = self.client.get_entity(
            id=entity_id,
        ).execute(
            action=action,
            params=params,
            text=text,
            connected_account_id=connected_account_id,
        )
        if self.output_in_file:
            return self._write_to_file(
                action=action,
                output=output,
                entity_id=entity_id,
            )
        try:
            # Save the variables of type file to the composio/output directory.
            output_modified = self._save_var_files(
                f"{action.name}_{entity_id}_{time.time()}", output
            )
            return output_modified
        except Exception:
            pass
        return output

    def _save_var_files(self, file_name_prefix: str, output: dict) -> dict:
        success_response_model = SuccessExecuteActionResponseModel.model_validate(
            output
        )
        resp_data = json.loads(success_response_model.response_data)
        for key, val in resp_data.items():
            try:
                file_model = FileModel.model_validate(val)
                _ensure_output_dir_exists()
                output_file_path = (
                    output_dir
                    / f"{file_name_prefix}_{file_model.name.replace('/', '_')}"
                )
                _write_file(output_file_path, base64.b64decode(file_model.content))
                resp_data[key] = str(output_file_path)
            except Exception:
                pass
        success_response_model.response_data = resp_data
        return success_response_model.model_dump()

    def _write_to_file(
        self,
        action: Action,
        output: t.Dict,
        entity_id: str = DEFAULT_ENTITY_ID,
    ) -> t.Dict:
        """Write output to a file."""
        filename = hashlib.sha256(
            f"{action.name}-{entity_id}-{time.time()}".encode()
        ).hexdigest()
        _ensure_output_dir_exists()
        outfile = output_dir / filename
        self.logger.info(f"Writing output to: {outfile}")
        _write_file(outfile, json.dumps(output))
        return {
            "message": f"output written to {outfile.resolve()}",
            "file": str(outfile.resolve()),
        }

    def execute_action(
        self,
        action: ActionType,
        params: dict,
        metadata: t.Optional[t.Dict] = None,
        entity_id: str = DEFAULT_ENTITY_ID,
        text: t.Optional[str] = None,
        connected_account_id: t.Optional[str] = None,
    ) -> t.Dict:
        action_start_time = time.time()
        action_log = {
            "action": str(action),
            "params": params,
            "start_time": action_start_time,
            "end_time": None,
            "duration": None,
            "status": "started",
            "error": None,
        }
        try:
            action = Action(action)
            if action.is_local:
                result = self._execute_local(
                    action=action,
                    params=params,
                    metadata=metadata,
                )
            else:
                result = self._execute_remote(
                    action=action,
                    params=params,
                    entity_id=entity_id,
                    text=text,
                    connected_account_id=connected_account_id,
                )
            action_log["status"] = "success"
            action_log["response"] = result
        except Exception as e:
            action_log["status"] = "failure"
            action_log["error"] = str(e)
            raise
        finally:
            action_log["end_time"] = time.time()
            action_log["duration"] = action_log["end_time"] - action_log["start_time"]
            try:
                self._log_action(action_log)
            except Exception as e:
                self.logger.error(f"Failed to log action: {action_log}. Error: {str(e)}")
        return result

    def _log_action(self, action_log: dict):
        log_dir_path = Path("action_logs")
        log_dir_path.mkdir(parents=True, exist_ok=True)

        if ComposioToolSet._run_start_timestamp is None:
            ComposioToolSet._run_start_timestamp = action_log["start_time"]

        log_file_path = log_dir_path / f"{ComposioToolSet._run_start_timestamp}.json"

        try:
            if log_file_path.exists():
                with open(log_file_path, "r") as file:
                    logs = json.load(file)
            else:
                logs = []

            if action_log.get("response", {}).get("status") == "failure":
                action_log["status"] = "tool_failure"
                action_log["error"] = action_log["response"].get("details")

            logs.append(action_log)

            with open(log_file_path, "w") as file:
                json.dump(logs, file, indent=4)
        except Exception as e:
            self.logger.error(f"Failed to write to log file: {log_file_path}. Error: {str(e)}")

    def get_action_schemas(
        self,
        apps: t.Optional[t.Sequence[AppType]] = None,
        actions: t.Optional[t.Sequence[ActionType]] = None,
        tags: t.Optional[t.Sequence[TagType]] = None,
    ) -> t.List[ActionModel]:
        runtime_actions = t.cast(
            t.List[t.Type[LocalAction]],
            [action for action in actions or [] if hasattr(action, "run_on_shell")],
        )
        actions = t.cast(
            t.List[Action],
            [
                Action(action)
                for action in actions or []
                if action not in runtime_actions
            ],
        )
        apps = t.cast(t.List[App], [App(app) for app in apps or []])

        local_actions = [action for action in actions if action.is_local]
        local_apps = [app for app in apps if app.is_local]

        remote_actions = [action for action in actions if not action.is_local]
        remote_apps = [app for app in apps if not app.is_local]

        items: t.List[ActionModel] = []
        if len(local_actions) > 0 or len(local_apps) > 0:
            items += [
                ActionModel(**item)
                for item in self._local_client.get_action_schemas(
                    apps=local_apps,
                    actions=local_actions,
                    tags=tags,
                )
            ]

        if len(remote_actions) > 0 or len(remote_apps) > 0:
            remote_items = self.client.actions.get(
                apps=remote_apps,
                actions=remote_actions,
                tags=tags,
            )
            items = items + remote_items

        for item in items:
            self.check_connected_account(action=item.name)
            item = self.action_preprocessing(item)
        items += [ActionModel(**act().get_action_schema()) for act in runtime_actions]
        return items

    def action_preprocessing(self, action_item: ActionModel) -> ActionModel:
        for param_name, param_details in action_item.parameters.properties.items():
            if param_details.get("properties") == FileModel.schema().get("properties"):
                action_item.parameters.properties[param_name].pop("properties")
                action_item.parameters.properties[param_name] = {
                    "type": "string",
                    "format": "file-path",
                    "description": f"File path to {param_details.get('description', '')}",
                }
            elif param_details.get("allOf", [{}])[0].get(
                "properties"
            ) == FileModel.schema().get("properties"):
                action_item.parameters.properties[param_name].pop("allOf")
                action_item.parameters.properties[param_name].update(
                    {
                        "type": "string",
                        "format": "file-path",
                        "description": f"File path to {param_details.get('description', '')}",
                    }
                )

        return action_item

    def create_trigger_listener(self, timeout: float = 15.0) -> TriggerSubscription:
        """Create trigger subscription."""
        return self.client.triggers.subscribe(timeout=timeout)

    def find_actions_by_use_case(
        self,
        *apps: AppType,
        use_case: str,
    ) -> t.List[Action]:
        """
        Find actions by specified use case.

        :param apps: List of apps to search.
        :param use_case: String describing the use case.
        :return: A list of actions matching the relevant use case.
        """
        actions = self.client.actions.get(
            apps=[App(app) for app in apps],
            use_case=use_case,
            allow_all=True,
        )
        return [
            Action(value=get_enum_key(name=action.name).lower()) for action in actions
        ]

    def find_actions_by_tags(
        self,
        *apps: AppType,
        tags: t.List[str],
    ) -> t.List[Action]:
        """
        Find actions by specified use case.

        :param apps: List of apps to search.
        :param use_case: String describing the use case.
        :return: A list of actions matching the relevant use case.
        """
        if len(tags) == 0:
            raise ComposioClientError(
                "Please provide at least one tag to perform search"
            )

        if len(apps) > 0:
            return list(
                itertools.chain(
                    *[list(App(app).get_actions(tags=tags)) for app in apps]
                )
            )

        actions = []
        for action in Action.all():
            if any(tag in action.tags for tag in tags):
                actions.append(action)
        return actions

    def get_agent_instructions(
        self,
        apps: t.Optional[t.Sequence[AppType]] = None,
        actions: t.Optional[t.Sequence[ActionType]] = None,
        tags: t.Optional[t.Sequence[TagType]] = None,
    ) -> str:
        """
        Generate a formatted string with instructions for agents based on the provided apps, actions, and tags.

        This function compiles a list of available tools from the specified apps, actions, and tags,
        and formats them into a human-readable string that can be used as instructions for agents.

        :param apps: Optional sequence of AppType to include in the search.
        :param actions: Optional sequence of ActionType to include in the search.
        :param tags: Optional sequence of TagType to filter the actions.
        :return: A formatted string with instructions for agents.
        """
        # Retrieve schema information for the given apps, actions, and tags
        schema_list = [
            schema.model_dump()
            for schema in (
                self.get_action_schemas(apps=apps, tags=tags)
                + self.get_action_schemas(actions=actions)
            )
        ]
        schema_info = [
            (schema_obj["appName"], schema_obj["name"]) for schema_obj in schema_list
        ]

        # Helper function to format a list of items into a string
        def format_list(items):
            if not items:
                return ""
            if len(items) == 1:
                return items[0]
            return ", ".join(items[:-2] + [" and ".join(items[-2:])])

        # Organize the schema information by app name
        action_dict: t.Dict[str, t.List] = {}
        for appName, name in schema_info:
            if appName not in action_dict:
                action_dict[appName] = []
            action_dict[appName].append(name)

        # Format the schema information into a human-readable string
        formatted_schema_info = (
            "You have various tools, among which "
            + ", ".join(
                [
                    f"for interacting with **{appName}** you might use {format_list(action_items)} tools"
                    for appName, action_items in action_dict.items()
                ]
            )
            + ". Whichever tool is useful to execute your task, use that with proper parameters."
        )
        return formatted_schema_info


def _ensure_output_dir_exists():
    """Ensure the output directory exists."""
    if not output_dir.exists():
        output_dir.mkdir()


def _write_file(file_path: t.Union[str, os.PathLike], content: t.Union[str, bytes]):
    """Write content to a file."""
    if isinstance(content, str):
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
    else:
        with open(file_path, "wb") as file:
            file.write(content)

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            log_dir = Path("action_logs")
            logs = ComposioToolSet()._load_logs(log_dir)
            df = ComposioToolSet()._prepare_data(logs)
            fig = px.bar(df, x="action", y="duration", color="status", barmode="group")
            graph_html = fig.to_html(full_html=False)

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(graph_html.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')