#!/usr/bin/env python3
"""
SensorScope Analytics Setup Script
Helps with initial project setup and configuration
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def print_header():
    print("=" * 60)
    print("📊 SensorScope Analytics - Setup Script")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")
        return True

def create_virtual_environment():
    """Create a virtual environment if it doesn't exist"""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("\n📦 Creating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            print("✅ Virtual environment created successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to create virtual environment")
            return False
    else:
        print("✅ Virtual environment already exists")
        return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📚 Installing dependencies...")
    
    # Determine the correct pip path
    if os.name == 'nt':  # Windows
        pip_path = Path("venv/Scripts/pip")
    else:  # macOS/Linux
        pip_path = Path("venv/bin/pip")
    
    try:
        subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("💡 Try running: pip install -r requirements.txt")
        return False

def setup_environment_file():
    """Setup environment configuration file"""
    env_path = Path(".env")
    template_path = Path(".env.template")
    
    if not env_path.exists() and template_path.exists():
        print("\n⚙️ Setting up environment configuration...")
        shutil.copy(template_path, env_path)
        print("✅ .env file created from template")
        print("💡 Please update the DATABASE_PATH in .env file")
        return True
    elif env_path.exists():
        print("✅ .env file already exists")
        return True
    else:
        print("⚠️ Environment template not found, creating basic .env file...")
        with open(env_path, 'w') as f:
            f.write("DATABASE_PATH=sensor_data.db\n")
        print("✅ Basic .env file created")
        return True

def check_database_file():
    """Check if database file exists"""
    print("\n🗄️ Checking database configuration...")
    
    # Try to read DATABASE_PATH from .env
    env_path = Path(".env")
    db_path = "sensor_data.db"  # default
    
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.startswith('DATABASE_PATH='):
                    db_path = line.split('=', 1)[1].strip()
                    break
    
    if Path(db_path).exists():
        print(f"✅ Database found: {db_path}")
        return True
    else:
        print(f"⚠️ Database not found: {db_path}")
        print("💡 Please place your SQLite database file in the project directory")
        print("💡 Or update the DATABASE_PATH in your .env file")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("🚀 Setup Complete! Next Steps:")
    print("=" * 60)
    print()
    print("1. Activate your virtual environment:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # macOS/Linux
        print("   source venv/bin/activate")
    print()
    print("2. Ensure your database file is properly configured in .env")
    print()
    print("3. Run the application:")
    print("   streamlit run app.py")
    print()
    print("4. Open your browser to: http://localhost:8501")
    print()
    print("📖 For more information, check the README.md file")
    print("👥 Team collaboration tips are available in the app")

def main():
    """Main setup function"""
    print_header()
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Create virtual environment
    if success and not create_virtual_environment():
        success = False
    
    # Install dependencies
    if success and not install_dependencies():
        success = False
    
    # Setup environment file
    if success:
        setup_environment_file()
    
    # Check database
    check_database_file()
    
    # Print next steps
    print_next_steps()
    
    if not success:
        print("\n❌ Setup completed with some issues. Please resolve them before running the app.")
        sys.exit(1)
    else:
        print("\n✅ Setup completed successfully!")

if __name__ == "__main__":
    main() 