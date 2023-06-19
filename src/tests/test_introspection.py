from dataclasses import dataclass
from typing import List, Optional

import pytest
from gpt_commands.openai.introspection import (
    UnsupportedDictionaryKeyTypeException,
    UnsupportedTypeException,
    create_manager,
    type_to_json_schema,
    decode_json,
)
from dataclasses_jsonschema import JsonSchemaMixin


@dataclass
class Point(JsonSchemaMixin):
    "A 2D point"
    x: float
    y: float


@dataclass
class Plane(JsonSchemaMixin):
    "A 2D plane"
    origin: Point
    normal: Point
    selected_points: List[Point]
    label_to_point: dict[str, Point]


@dataclass
class DataclassWithoutMixin:
    x: float
    y: float


class NonDataclass:
    x: float
    y: float


def test_type_to_json_schema():
    # Test string type
    assert type_to_json_schema(str) == {"type": "string"}

    # Test integer type
    assert type_to_json_schema(int) == {"type": "integer"}

    # Test float type
    assert type_to_json_schema(float) == {"type": "number"}

    # Test boolean type
    assert type_to_json_schema(bool) == {"type": "boolean"}

    # Test list type
    assert type_to_json_schema(list[int]) == {
        "type": "array",
        "items": {"type": "integer"},
    }

    # Test nested list type
    assert type_to_json_schema(list[list[int]]) == {
        "type": "array",
        "items": {"type": "array", "items": {"type": "integer"}},
    }

    # Test dictionary type
    assert type_to_json_schema(dict[str, int]) == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }

    # Test nested dictionary type
    assert type_to_json_schema(dict[str, dict[str, list[int]]]) == {
        "type": "object",
        "additionalProperties": {
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "integer"}},
        },
    }

    # Test unsupported dictionary key type
    with pytest.raises(
        UnsupportedDictionaryKeyTypeException,
        match="Unsupported dictionary key type: int. Only string keys are supported.",
    ):
        type_to_json_schema(dict[int, str])

    # Test unsupported type
    with pytest.raises(
        UnsupportedTypeException,
        match="Unsupported type: set. Only primitive types, lists, dictionaries, Optional and JsonSchemaMixin dataclasses are supported.",
    ):
        type_to_json_schema(set)

    # Test dataclass
    assert type_to_json_schema(Point) == {
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
        },
        "required": ["x", "y"],
        "description": "A 2D point",
    }

    # Test nested dataclass
    assert type_to_json_schema(Plane) == {
        "type": "object",
        "required": ["origin", "normal", "selected_points", "label_to_point"],
        "properties": {
            "origin": {"$ref": "#/definitions/Point"},
            "normal": {"$ref": "#/definitions/Point"},
            "selected_points": {
                "type": "array",
                "items": {"$ref": "#/definitions/Point"},
            },
            "label_to_point": {
                "type": "object",
                "additionalProperties": {"$ref": "#/definitions/Point"},
            },
        },
        "description": "A 2D plane",
        "definitions": {
            "Point": {
                "type": "object",
                "required": ["x", "y"],
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                "description": "A 2D point",
            }
        },
    }

    # Test unsupported dataclass
    with pytest.raises(
        UnsupportedTypeException,
        match="Unsupported type: DataclassWithoutMixin. Only primitive types, lists, dictionaries, Optional and JsonSchemaMixin dataclasses are supported.",
    ):
        type_to_json_schema(DataclassWithoutMixin)

    # Test unsupported class
    with pytest.raises(
        UnsupportedTypeException,
        match="Unsupported type: NonDataclass. Only primitive types, lists, dictionaries, Optional and JsonSchemaMixin dataclasses are supported.",
    ):
        type_to_json_schema(NonDataclass)

    # Test optional type
    assert type_to_json_schema(Optional[int]) == {"type": "integer"}
    assert type_to_json_schema(Optional[Point]) == {
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
        },
        "required": ["x", "y"],
        "description": "A 2D point",
    }


def test_decode_json():
    # Test string type
    assert decode_json("hello", str) == "hello"
    assert decode_json('"hello"', str) == "hello"
    assert decode_json("123", str) == "123"
    assert decode_json("123.456", str) == "123.456"
    assert decode_json("true", str) == "true"

    # Test integer type
    assert decode_json("123", int) == 123

    # Test float type
    assert decode_json("123.456", float) == 123.456

    # Test boolean type
    assert decode_json("true", bool) == True
    assert decode_json("false", bool) == False

    # Test list type
    assert decode_json("[1, 2, 3]", list[int]) == [1, 2, 3]
    assert decode_json("[1, 2, 3]", list[float]) == [1, 2, 3]
    assert decode_json('["1", "2", "3"]', list[str]) == ["1", "2", "3"]

    # Test nested list type
    assert decode_json("[[1, 2], [3, 4]]", list[list[int]]) == [[1, 2], [3, 4]]
    assert decode_json("[[1, 2], [3, 4]]", list[list[float]]) == [[1, 2], [3, 4]]
    assert decode_json('[["1", "2"], ["3", "4"]]', list[list[str]]) == [
        ["1", "2"],
        ["3", "4"],
    ]

    # Test dictionary type
    assert decode_json('{"a": 1, "b": 2}', dict[str, int]) == {"a": 1, "b": 2}
    assert decode_json('{"a": 1, "b": 2}', dict[str, float]) == {"a": 1, "b": 2}
    assert decode_json('{"a": "1", "b": "2"}', dict[str, str]) == {"a": "1", "b": "2"}
    assert decode_json('{"a": true, "b": false}', dict[str, bool]) == {
        "a": True,
        "b": False,
    }
    assert decode_json('{"a": [1,2,3], "b": [4,5,6]}', dict[str, list[int]]) == {
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    }

    # Test nested dictionary type
    assert decode_json(
        '{"a": {"b": [1,2,3]}, "c": {"d": [4,5,6]}}', dict[str, dict[str, list[int]]]
    ) == {"a": {"b": [1, 2, 3]}, "c": {"d": [4, 5, 6]}}
    assert decode_json(
        '{"a": {"b": [1,2,3]}, "c": {"d": [4,5,6]}}', dict[str, dict[str, list[float]]]
    ) == {"a": {"b": [1, 2, 3]}, "c": {"d": [4, 5, 6]}}
    assert decode_json(
        '{"a": {"b": ["1","2","3"]}, "c": {"d": ["4","5","6"]}}',
        dict[str, dict[str, list[str]]],
    ) == {"a": {"b": ["1", "2", "3"]}, "c": {"d": ["4", "5", "6"]}}

    # Test dataclass type
    assert decode_json('{"x": 1, "y": 2}', Point) == Point(x=1, y=2)

    # Test nested dataclass type
    assert (
        decode_json(
            """
        {
            "origin": {"x": 1, "y": 2}, 
            "normal": {"x": 3, "y": 4}, 
            "selected_points": [{"x": 5, "y": 6}, {"x": 7, "y": 8}], 
            "label_to_point": {"a": {"x": 9, "y": 10}, "b": {"x": 11, "y": 12}}
        }
        """,
            Plane,
        )
        == Plane(
            origin=Point(x=1, y=2),
            normal=Point(x=3, y=4),
            selected_points=[Point(x=5, y=6), Point(x=7, y=8)],
            label_to_point={"a": Point(x=9, y=10), "b": Point(x=11, y=12)},
        )
    )

    # Test optional type
    assert decode_json("null", Optional[int]) == None
    assert decode_json("null", Optional[float]) == None
    assert decode_json("null", Optional[str]) == None
    assert decode_json("null", Optional[bool]) == None
    assert decode_json("null", Optional[list[int]]) == None
    assert decode_json("null", Optional[list[float]]) == None
    assert decode_json("null", Optional[list[str]]) == None
    assert decode_json("null", Optional[dict[str, int]]) == None
    assert decode_json("null", Optional[dict[str, float]]) == None
    assert decode_json("null", Optional[dict[str, str]]) == None
    assert decode_json("null", Optional[dict[str, bool]]) == None
    assert decode_json("null", Optional[dict[str, list[int]]]) == None
    assert decode_json("null", Optional[dict[str, list[float]]]) == None
    assert decode_json("null", Optional[dict[str, list[str]]]) == None
    assert decode_json("null", Optional[Point]) == None
    assert decode_json("null", Optional[Plane]) == None
    assert decode_json("asdf", Optional[str]) == "asdf"
    assert decode_json("123", Optional[int]) == 123
    assert decode_json("123.456", Optional[float]) == 123.456
    assert decode_json("true", Optional[bool]) == True
    assert decode_json("false", Optional[bool]) == False
    assert decode_json("[1,2,3]", Optional[list[int]]) == [1, 2, 3]
    assert decode_json("[1,2,3]", Optional[list[float]]) == [1, 2, 3]
    assert decode_json('["1","2","3"]', Optional[list[str]]) == ["1", "2", "3"]
    assert decode_json('{"a": 1, "b": 2}', Optional[dict[str, int]]) == {"a": 1, "b": 2}
    assert decode_json('{"a": 1, "b": 2}', Optional[dict[str, float]]) == {
        "a": 1,
        "b": 2,
    }
    assert decode_json('{"a": "1", "b": "2"}', Optional[dict[str, str]]) == {
        "a": "1",
        "b": "2",
    }
    assert decode_json('{"a": true, "b": false}', Optional[dict[str, bool]]) == {
        "a": True,
        "b": False,
    }
    assert decode_json(
        '{"a": [1,2,3], "b": [4,5,6]}', Optional[dict[str, list[int]]]
    ) == {
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    }
    assert decode_json(
        '{"a": [1,2,3], "b": [4,5,6]}', Optional[dict[str, list[float]]]
    ) == {"a": [1, 2, 3], "b": [4, 5, 6]}
    assert decode_json(
        '{"a": ["1","2","3"], "b": ["4","5","6"]}', Optional[dict[str, list[str]]]
    ) == {"a": ["1", "2", "3"], "b": ["4", "5", "6"]}
    assert decode_json('{"x": 1, "y": 2}', Optional[Point]) == Point(x=1, y=2)
    assert (
        decode_json(
            """
        {
            "origin": {"x": 1, "y": 2},
            "normal": {"x": 3, "y": 4},
            "selected_points": [{"x": 5, "y": 6}, {"x": 7, "y": 8}],
            "label_to_point": {"a": {"x": 9, "y": 10}, "b": {"x": 11, "y": 12}}
        }
        """,
            Optional[Plane],
        )
        == Plane(
            origin=Point(x=1, y=2),
            normal=Point(x=3, y=4),
            selected_points=[Point(x=5, y=6), Point(x=7, y=8)],
            label_to_point={"a": Point(x=9, y=10), "b": Point(x=11, y=12)},
        )
    )


def test_primitives():
    class ClassWithPrimitivesOnly:
        def get_stuff(
            self, text: str, number: int, float_number: float, flag: bool
        ) -> List[str]:
            """
            Gets stuff

            Args:
                text (str): Sample text
                number (int): Sample number
                float_number (float): Sample float number
                flag (bool): Sample flag

            Returns:
                List[str]: Sample list of strings
            """
            return []

    manager = create_manager(ClassWithPrimitivesOnly())

    assert len(manager.functions) == 1
    function = list(manager.functions.values())[0]
    assert function.name == "get_stuff"
    assert function.description == "Gets stuff"
    assert function.has_return

    schema = function.json_schema()
    assert schema == {
        "name": "get_stuff",
        "description": "Gets stuff",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Sample text"},
                "number": {"type": "integer", "description": "Sample number"},
                "float_number": {
                    "type": "number",
                    "description": "Sample float number",
                },
                "flag": {"type": "boolean", "description": "Sample flag"},
            },
            "required": ["text", "number", "float_number", "flag"],
        },
    }


def test_list():
    class ClassWithList:
        def get_stuff(self, list_of_stuff: List[str]) -> List[str]:
            """
            Gets stuff

            Args:
                list_of_stuff (List[str]): Sample list of strings

            Returns:
                List[str]: Sample list of strings
            """
            return []

    manager = create_manager(ClassWithList())

    assert len(manager.functions) == 1
    function = list(manager.functions.values())[0]
    assert function.name == "get_stuff"
    assert function.description == "Gets stuff"
    assert function.has_return

    schema = function.json_schema()
    assert schema == {
        "name": "get_stuff",
        "description": "Gets stuff",
        "parameters": {
            "type": "object",
            "properties": {
                "list_of_stuff": {
                    "description": "Sample list of strings",
                    "items": {"type": "string"},
                    "type": "array",
                }
            },
            "required": ["list_of_stuff"],
        },
    }


def test_custom_objects():
    class ClassWithCustomObjects:
        def get_stuff(
            self, planes: List[Plane], point: Point, markers: dict[str, Point]
        ) -> List[str]:
            """
            Gets stuff

            Args:
                planes (List[Plane]): Sample list of planes
                point (Point): Sample point
                markers (dict[str, Point]): Sample dictionary of markers

            Returns:
                List[str]: Sample list of strings
            """
            return []

    manager = create_manager(ClassWithCustomObjects())
    assert len(manager.functions) == 1
    function = list(manager.functions.values())[0]
    assert function.name == "get_stuff"
    assert function.description == "Gets stuff"
    assert function.has_return

    schema = function.json_schema()
    assert schema == {
        "name": "get_stuff",
        "description": "Gets stuff",
        "parameters": {
            "type": "object",
            "properties": {
                "planes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "origin",
                            "normal",
                            "selected_points",
                            "label_to_point",
                        ],
                        "properties": {
                            "origin": {"$ref": "#/definitions/Point"},
                            "normal": {"$ref": "#/definitions/Point"},
                            "selected_points": {
                                "type": "array",
                                "items": {"$ref": "#/definitions/Point"},
                            },
                            "label_to_point": {
                                "type": "object",
                                "additionalProperties": {"$ref": "#/definitions/Point"},
                            },
                        },
                        "description": "A 2D plane",
                        "definitions": {
                            "Point": {
                                "type": "object",
                                "required": ["x", "y"],
                                "properties": {
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                },
                                "description": "A 2D point",
                            }
                        },
                    },
                    "description": "Sample list of planes",
                },
                "point": {
                    "type": "object",
                    "required": ["x", "y"],
                    "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                    "description": "Sample point",
                },
                "markers": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "required": ["x", "y"],
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                        },
                        "description": "A 2D point",
                    },
                    "description": "Sample dictionary of markers",
                },
            },
            "required": ["planes", "point", "markers"],
        },
    }


def test_optional():
    class ClassWithOptional:
        def get_stuff(self, optional: Optional[str] = None) -> Optional[str]:
            """
            Gets stuff

            Args:
                optional (Optional[str]): Sample optional string

            Returns:
                Optional[str]: Sample optional string
            """
            return None

    manager = create_manager(ClassWithOptional())
    assert len(manager.functions) == 1
    function = list(manager.functions.values())[0]
    assert function.name == "get_stuff"
    assert function.description == "Gets stuff"
    assert function.has_return

    schema = function.json_schema()
    assert schema == {
        "name": "get_stuff",
        "description": "Gets stuff",
        "parameters": {
            "type": "object",
            "properties": {
                "optional": {
                    "type": "string",
                    "description": "Sample optional string",
                }
            },
            "required": [],
        },
    }


def test_no_return():
    class ClassWithVoid:
        def get_stuff(self, simple: str) -> None:
            """
            Gets stuff

            Args:
                simple (str): Sample string
            """
            pass

    manager = create_manager(ClassWithVoid())
    assert len(manager.functions) == 1
    function = list(manager.functions.values())[0]
    assert function.name == "get_stuff"
    assert function.description == "Gets stuff"
    assert not function.has_return

    schema = function.json_schema()
    assert schema == {
        "name": "get_stuff",
        "description": "Gets stuff",
        "parameters": {
            "type": "object",
            "properties": {
                "simple": {"type": "string", "description": "Sample string"}
            },
            "required": ["simple"],
        },
    }


def test_no_return_hint():
    class ClassWithoutReturnType:
        def get_stuff(self, simple: str):
            """
            Gets stuff

            Args:
                simple (str): Sample string
            """
            pass

    manager = create_manager(ClassWithoutReturnType())
    assert len(manager.functions) == 1
    function = list(manager.functions.values())[0]
    assert function.name == "get_stuff"
    assert function.description == "Gets stuff"
    assert not function.has_return

    schema = function.json_schema()
    assert schema == {
        "name": "get_stuff",
        "description": "Gets stuff",
        "parameters": {
            "type": "object",
            "properties": {
                "simple": {"type": "string", "description": "Sample string"}
            },
            "required": ["simple"],
        },
    }


def test_missing_hints():
    class ClassWithoutHints:
        def get_stuff(self, planes):
            """
            Gets stuff

            Args:
                planes (List[Plane]): Sample list of planes

            Returns:
                List[str]: Sample list of strings
            """
            return []

    with pytest.raises(
        Exception,
        match="Missing type hint for parameter planes in function get_stuff",
    ):
        create_manager(ClassWithoutHints())


def test_missing_method_docstring():
    class ClassWithoutDocstring:
        def get_stuff(self, planes: List[str]):
            return []

    with pytest.raises(
        Exception,
        match="Missing docstring for function get_stuff",
    ):
        create_manager(ClassWithoutDocstring())


def test_missing_parameter_docstring():
    class ClassWithoutHints:
        def get_stuff(self, simple: str, planes: List[str]):
            """
            Gets stuff

            Args:
                simple (str): Sample string
            """
            return []

    with pytest.raises(
        Exception,
        match="Missing docstring for parameter planes in function get_stuff",
    ):
        create_manager(ClassWithoutHints())


def test_execution():
    class ClassToTest:
        def get_stuff(self, simple: str) -> str:
            """
            Gets stuff

            Args:
                simple (str): Sample string

            Returns:
                str: Sample string
            """
            return simple + "123"
        
    
    manager = create_manager(ClassToTest())
    assert len(manager.functions) == 1
    function = list(manager.functions.values())[0]
    assert function.name == "get_stuff"
    assert function.description == "Gets stuff"
    assert function.has_return

    result = manager.execute("get_stuff", {"simple": "test"})
    assert result == '"test123"'

def test_execution_with_optional():
    class ClassToTest:
        def get_stuff(self, simple: str, optional: Optional[int] = 123) -> str:
            """
            Gets stuff

            Args:
                simple (str): Sample string
                optional (Optional[str]): Sample optional string

            Returns:
                str: Sample string
            """
            return f"{simple}{optional}"
        
    
    manager = create_manager(ClassToTest())
    assert len(manager.functions) == 1
    function = list(manager.functions.values())[0]
    assert function.name == "get_stuff"
    assert function.description == "Gets stuff"
    assert function.has_return

    result = manager.execute("get_stuff", {"simple": "test"})
    assert result == '"test123"'
    result = manager.execute("get_stuff", {"simple": "test", "optional": 456})
    assert result == '"test456"'
    result = manager.execute("get_stuff", {"simple": "test", "optional": None})
    assert result == '"testNone"'
