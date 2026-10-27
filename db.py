"""
MySQL storage for the Smart Online Attendance System.

Connection settings are read from, in this order:
  1. Streamlit secrets, section [mysql] (used when deployed on Streamlit Cloud)
  2. Environment variables MYSQL_HOST, MYSQL_PORT, MYSQL_USER,
     MYSQL_PASSWORD, MYSQL_DATABASE (used when running attendance.py locally)

Each student is stored once per day (UNIQUE on name + date).
"""
import os

import mysql.connector
from mysql.connector import errorcode

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    att_date DATE NOT NULL,
    att_time TIME NOT NULL,
    UNIQUE KEY uniq_person_day (name, att_date)
)
"""


def _config():
    try:
        import streamlit as st
        if "mysql" in st.secrets:
            cfg = dict(st.secrets["mysql"])
            cfg["port"] = int(cfg.get("port", 3306))
            return cfg
    except Exception:
        pass
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "attendance_db"),
    }


def get_connection():
    cfg = _config()
    try:
        return mysql.connector.connect(connection_timeout=10, **cfg)
    except mysql.connector.Error as err:
        if err.errno != errorcode.ER_BAD_DB_ERROR:
            raise
        # The database does not exist yet: create it, then connect again.
        database = cfg.pop("database")
        admin = mysql.connector.connect(connection_timeout=10, **cfg)
        cur = admin.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
        admin.commit()
        cur.close()
        admin.close()
        return mysql.connector.connect(connection_timeout=10, database=database, **cfg)


def init_db():
    """Create the attendance table if needed. Raises if MySQL is unreachable."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(TABLE_SQL)
        conn.commit()
        cur.close()
    finally:
        conn.close()


def mark_attendance(name):
    """Record the student for today. Returns True if newly marked, False if already marked."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT IGNORE INTO attendance (name, att_date, att_time) "
            "VALUES (%s, CURDATE(), CURTIME())",
            (name,),
        )
        conn.commit()
        newly_marked = cur.rowcount == 1
        cur.close()
        return newly_marked
    finally:
        conn.close()


def fetch_today():
    """Today's attendance as a list of dicts."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name, att_date, att_time FROM attendance "
            "WHERE att_date = CURDATE() ORDER BY att_time"
        )
        rows = [
            {"Name": n, "Date": str(d), "Time": str(t)} for n, d, t in cur.fetchall()
        ]
        cur.close()
        return rows
    finally:
        conn.close()
