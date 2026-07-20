from dataclasses import dataclass, field
from uuid import UUID, uuid4

@dataclass
class Entity:
     id: UUID = field(default_factory=uuid4)
     name: str = ""

@dataclass
class Person(Entity):
      pass
@dataclass
class Animal(Entity):
      pass
@dataclass
class Place(Entity):
      pass
@dataclass
class PhysicalObject(Entity):
      pass
@dataclass
class Organization(Entity):
      pass
@dataclass
class Event(Entity):
      pass
@dataclass
class Concept(Entity):
      pass
@dataclass
class Document(Entity):
      pass
