"""Build a small example SQLite database for trying PV-SQL.

    python examples/build_example_db.py

Creates `examples/university.sqlite`. No download, no benchmark data needed.

The schema deliberately contains the kinds of traps that make schema-only
text-to-SQL fail, so the probing stage has something to actually discover:

  * `students.status` holds 'active'/'withdrawn'/'on_leave' -- not the wording
    a question would use ("currently enrolled")
  * `students.enrolled_on` is TEXT in 'YYYY-MM-DD' form, not a numeric year
  * `enrollments.grade` is a letter grade ('A', 'A-', 'B+'), so averaging it
    requires a mapping rather than AVG()
  * `students.gpa` is nullable, which breaks naive ORDER BY ... DESC LIMIT 1
  * department names are spelled out but joined through integer ids
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "university.sqlite"

SCHEMA = """
CREATE TABLE departments (
    dept_id   INTEGER PRIMARY KEY,
    dept_name TEXT NOT NULL,
    building  TEXT
);

CREATE TABLE students (
    student_id  INTEGER PRIMARY KEY,
    full_name   TEXT NOT NULL,
    dept_id     INTEGER,
    enrolled_on TEXT,            -- 'YYYY-MM-DD'
    gpa         REAL,            -- nullable: first-term students have none yet
    status      TEXT,            -- 'active' | 'withdrawn' | 'on_leave'
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
);

CREATE TABLE courses (
    course_id INTEGER PRIMARY KEY,
    title     TEXT NOT NULL,
    dept_id   INTEGER,
    credits   INTEGER,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
);

CREATE TABLE enrollments (
    enroll_id  INTEGER PRIMARY KEY,
    student_id INTEGER,
    course_id  INTEGER,
    term       TEXT,             -- '2023-Fall', '2024-Spring', ...
    grade      TEXT,             -- 'A', 'A-', 'B+', ... NULL while in progress
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (course_id)  REFERENCES courses(course_id)
);
"""

DEPARTMENTS = [
    (1, "Computer Science", "Turing Hall"),
    (2, "Physics", "Bohr Building"),
    (3, "Mathematics", "Noether Hall"),
    (4, "Economics", "Keynes Center"),
]

STUDENTS = [
    (1, "Alice Chen", 1, "2021-09-01", 3.9, "active"),
    (2, "Bob Martinez", 1, "2021-09-01", 3.2, "active"),
    (3, "Carla Reed", 2, "2022-09-01", 3.7, "active"),
    (4, "Daniel Okafor", 2, "2022-09-01", None, "active"),
    (5, "Elena Rossi", 3, "2020-09-01", 3.95, "withdrawn"),
    (6, "Farid Haddad", 3, "2023-09-01", 2.8, "active"),
    (7, "Grace Kim", 1, "2023-09-01", None, "on_leave"),
    (8, "Hugo Silva", 4, "2020-09-01", 3.45, "active"),
    (9, "Ivy Nakamura", 4, "2022-09-01", 3.6, "active"),
    (10, "Jonas Weber", 2, "2023-09-01", 3.1, "withdrawn"),
]

COURSES = [
    (101, "Introduction to Algorithms", 1, 4),
    (102, "Database Systems", 1, 3),
    (103, "Quantum Mechanics", 2, 4),
    (104, "Classical Dynamics", 2, 3),
    (105, "Real Analysis", 3, 4),
    (106, "Linear Algebra", 3, 3),
    (107, "Microeconomics", 4, 3),
]

ENROLLMENTS = [
    (1, 1, 101, "2023-Fall", "A"),
    (2, 1, 102, "2024-Spring", "A-"),
    (3, 2, 101, "2023-Fall", "B+"),
    (4, 2, 102, "2024-Spring", "B"),
    (5, 3, 103, "2023-Fall", "A"),
    (6, 3, 104, "2024-Spring", "A-"),
    (7, 4, 103, "2023-Fall", "B"),
    (8, 4, 106, "2024-Spring", None),
    (9, 5, 105, "2022-Fall", "A"),
    (10, 6, 105, "2023-Fall", "C+"),
    (11, 6, 106, "2024-Spring", "B-"),
    (12, 8, 107, "2023-Fall", "A-"),
    (13, 9, 107, "2023-Fall", "B+"),
    (14, 9, 106, "2024-Spring", None),
    (15, 10, 104, "2023-Fall", "C"),
    (16, 1, 106, "2024-Spring", "A"),
    (17, 3, 101, "2024-Spring", "B+"),
]


def build(db_path: Path = DB_PATH) -> Path:
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO departments VALUES (?,?,?)", DEPARTMENTS)
        conn.executemany("INSERT INTO students VALUES (?,?,?,?,?,?)", STUDENTS)
        conn.executemany("INSERT INTO courses VALUES (?,?,?,?)", COURSES)
        conn.executemany("INSERT INTO enrollments VALUES (?,?,?,?,?)", ENROLLMENTS)
        conn.commit()
    finally:
        conn.close()

    return db_path


if __name__ == "__main__":
    path = build()
    print(f"Created {path}")
    print(f"  {len(DEPARTMENTS)} departments, {len(STUDENTS)} students, "
          f"{len(COURSES)} courses, {len(ENROLLMENTS)} enrollments")
