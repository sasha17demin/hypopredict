"""
Type-safe data structures for hypopredict.
"""
from dataclasses import dataclass


@dataclass(frozen=True) # immutabkle onject because use as ID
class PersonDay:
    """
    Type-safe representation of a person-day combination.

    Replaces fragile integer encoding (e.g., 35 = person 3, day 5).

    Attributes:
        person_id: Integer person identifier (1-9)
        day: Integer day identifier (1-6)
    """
    # init will be done by dataclass
    person_id: int
    day: int

    def __post_init__(self):
        """Validate person_id and day ranges."""
        if not 1 <= self.person_id <= 9:
            raise ValueError(f"person_id must be between 1 and 9, got {self.person_id}")
        if not 1 <= self.day <= 4:
            raise ValueError(f"day must be between 1 and 4, got {self.day}")

    @classmethod
    def from_legacy_id(cls, legacy_id: int) -> "PersonDay":
        """
        Create PersonDay from legacy integer encoding.

        Args:
            legacy_id: Integer like 35 (person 3, day 5)

        Returns:
            PersonDay instance

        Example:
            >>> PersonDay.from_legacy_id(35)
            PersonDay(person_id=3, day=5)
        """
        person_id = legacy_id // 10
        day = legacy_id % 10
        return cls(person_id=person_id, day=day)

    def to_legacy_id(self) -> int:
        """
        Convert to legacy integer encoding.

        Returns:
            Integer like 35 (person 3, day 5)

        Example:
            >>> PersonDay(3, 5).to_legacy_id()
            35
        """
        return self.person_id * 10 + self.day

    def __str__(self) -> str: # what print(instance) would show
        """String representation."""
        return f"Person{self.person_id}_Day{self.day}"

    def __repr__(self) -> str: # developer representation -- shows the constructor
        """Developer representation."""
        return f"PersonDay(person_id={self.person_id}, day={self.day})"
