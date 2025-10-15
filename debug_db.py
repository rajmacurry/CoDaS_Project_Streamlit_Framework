#!/usr/bin/env python3
"""
Database Debug Script - Test database connection
"""

import os
import sqlite3
from dotenv import load_dotenv

def test_database_connection():
    """Test database connection and list tables"""
    load_dotenv()
    
    db_path = os.getenv('DATABASE_PATH', 'sensor_data.db')
    print(f"Testing database connection...")
    print(f"Database path: {db_path}")
    print(f"File exists: {os.path.exists(db_path)}")
    
    if not os.path.exists(db_path):
        print("❌ Database file not found!")
        print("Please create a .env file with DATABASE_PATH pointing to your SQLite database")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"✅ Connected successfully!")
        print(f"Found {len(tables)} tables:")
        
        for i, table in enumerate(tables, 1):
            table_name = table[0]
            print(f"{i}. {table_name}")
            
            # Get table info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            print(f"   Columns: {len(columns)}")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"   Rows: {count}")
            print()
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_database_connection() 