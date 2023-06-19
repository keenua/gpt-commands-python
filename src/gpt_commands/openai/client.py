from enum import Enum
from logging import getLogger
from os import getenv
import json
from typing import AsyncGenerator, List, Optional

from dotenv import load_dotenv
from aiohttp import ClientSession, StreamReader
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from gpt_commands.openai.introspection import Manager, create_manager


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"

    def __deepcopy__(self, memo):
        return self.value


@dataclass
class Message:
    role: Role
    content: str
    name: Optional[str] = None

    def to_request(self) -> dict:
        result = {
            "role": self.role.value,
            "content": self.content,
        }

        if self.name:
            result["name"] = self.name

        return result


@dataclass_json
@dataclass(frozen=True)
class FunctionCall:
    name: Optional[str] = None
    arguments: Optional[str] = None


@dataclass
class FunctionExecution:
    name: str
    arguments: dict[str, str]

    def execute(self, manager: Manager) -> Optional[str]:
        return manager.execute(self.name, self.arguments)
    
    def has_return(self, manager: Manager) -> bool:
        function = manager.get_function(self.name)
        if function is None:
            return False
        return function.has_return


@dataclass_json
@dataclass(frozen=True)
class Delta:
    role: Optional[Role] = None
    content: Optional[str] = None
    function_call: Optional[FunctionCall] = None


@dataclass_json
@dataclass(frozen=True)
class ChatCompletionChoice:
    index: int
    delta: Delta
    finish_reason: Optional[str]


@dataclass_json
@dataclass(frozen=True)
class ChatCompletionChunk:
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]


@dataclass
class ResponseData:
    ready: bool
    content: str
    function_name: Optional[str]
    function_arguments: str
    delta_text: Optional[str]

    def get_message(self) -> Optional[Message]:
        if self.ready and self.content:
            return Message(Role.ASSISTANT, self.content)
        else:
            return None

    def get_function_execution(self) -> Optional[FunctionExecution]:
        if self.ready and self.function_name and self.function_arguments:
            arguments = {key: json.dumps(value) for key, value in json.loads(self.function_arguments).items()}
            return FunctionExecution(self.function_name, arguments)
        else:
            return None


class GPTCommandsClient:
    def __init__(
        self,
        model: str,
        system_prompt: str,
        api_key: Optional[str] = None,
        api_organization: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ):
        load_dotenv()
        self.logger = getLogger(__name__)

        self.api_key = api_key or getenv("OPENAI_API_KEY")
        self.organization = api_organization or getenv("OPENAI_ORGANIZATION")
        self.messages: List[Message] = [Message(Role.SYSTEM, system_prompt)]
        self.session: Optional[ClientSession] = None

        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def __aenter__(self):
        self.session = ClientSession(
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    def __parse_stream_helper(self, line: bytes) -> Optional[str]:
        if line:
            if line.strip() == b"data: [DONE]":
                return None
            if line.startswith(b"data: "):
                line = line[len(b"data: ") :]
                return line.decode("utf-8")
            else:
                return None
        return None

    async def __parse_stream_async(self, rbody: StreamReader):
        async for line in rbody:
            _line = self.__parse_stream_helper(line)
            if _line is not None:
                yield _line

    def __process_chunk(
        self, chunk: ChatCompletionChunk, response: Optional[ResponseData]
    ) -> ResponseData:
        delta = chunk.choices[0].delta

        ready = bool(chunk.choices[0].finish_reason)

        text: Optional[str] = None
        content = response.content if response else ""

        if delta.content:
            text = delta.content.rstrip("\n")
            content += text

        function_name = response.function_name if response else None
        function_arguments = response.function_arguments if response else ""

        if delta.function_call:
            if delta.function_call.name:
                function_name = delta.function_call.name

            if delta.function_call.arguments:
                function_arguments += delta.function_call.arguments

        return ResponseData(
            ready=ready,
            content=content,
            function_name=function_name,
            function_arguments=function_arguments,
            delta_text=text,
        )

    async def __send_message(
        self, message_to_send: Message, manager: object
    ) -> AsyncGenerator[str, None]:
        self.messages.append(message_to_send)

        manager_wrapper = create_manager(manager)
        messages = [message.to_request() for message in self.messages]

        body = {
            "model": self.model,
            "messages": messages,
            "functions": [function.json_schema() for function in manager_wrapper.functions.values()],
            "max_tokens": self.max_tokens,
            "n": 1,
            "temperature": self.temperature,
            "stream": True,
        }
        response: Optional[ResponseData] = None

        if self.session:
            async with self.session.post(
                "https://api.openai.com/v1/chat/completions", json=body
            ) as resp:
                if resp.status != 200:
                    raise Exception(
                        f"OpenAI API returned status code {resp.status}: {await resp.text()}"
                    )
                async for data in self.__parse_stream_async(resp.content):
                    json_data = json.loads(data)

                    if json_data.get("error"):
                        raise Exception(f"OpenAI API returned error: {json_data['error']}")

                    chunk: ChatCompletionChunk = ChatCompletionChunk.from_dict(json_data)  # type: ignore
                    processed_chunk = self.__process_chunk(chunk, response)
                    if processed_chunk.delta_text:
                        yield processed_chunk.delta_text

                    response = processed_chunk
                    if response.ready:
                        break

        if response and response.ready:
            message = response.get_message()

            if message:
                self.logger.info(f"Storing message: {message}")
                self.messages.append(message)

            call = response.get_function_execution()

            if call:
                self.logger.info(f"Calling function: {call}")
                result = call.execute(manager_wrapper)
                if call.has_return(manager_wrapper):
                    function_result = Message(Role.FUNCTION, result or "null", call.name)
                    async for data in self.__send_message(function_result, manager):
                        yield data

    async def chat_stream(self, prompt: str, manager: object) -> AsyncGenerator[str, None]:
        """
        Sends a prompt to the OpenAI API and yields the response in chunks

        Args:
            prompt: User prompt to send to the API
            manager: The manager object to use for function calls
        
        Yields:
            The response from the API in string chunks
        """
        message = Message(Role.USER, prompt)
        async for data in self.__send_message(message, manager):
            yield data

    async def chat(self, prompt: str, manager: object) -> str:
        """
        Sends a prompt to the OpenAI API and returns the response as a string

        Args:
            prompt: The prompt to send to the API
            manager: The manager object to use for function calls

        Returns:
            The response from the API as a string
        """
        result = ""
        async for data in self.chat_stream(prompt, manager):
            result += data
        return result
