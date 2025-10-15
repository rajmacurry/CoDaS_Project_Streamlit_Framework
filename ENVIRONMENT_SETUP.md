# Environment Setup Guide

## Creating Your .env File

Since environment files are protected by `.gitignore`, you'll need to create your own `.env` file for database configuration.

### Step 1: Create the .env file

Create a new file named `.env` in the project root directory with the following content:

```env
# Database Configuration
DATABASE_PATH=sensor_data.db

# Optional: Database connection settings  
DB_TIMEOUT=30
DB_CHECK_SAME_THREAD=False
```

### Step 2: Update Database Path

Replace `sensor_data.db` with the actual path to your SQLite database file:

**Examples:**
```env
# If your database is in the project root:
DATABASE_PATH=my_sensor_data.db

# If your database is in a subdirectory:
DATABASE_PATH=data/sensors/production_data.db

# If your database is in an absolute path:
DATABASE_PATH=/Users/yourname/data/sensor_readings.db
```

### Step 3: Verify Configuration

1. Make sure your database file exists at the specified path
2. Ensure the file has proper read permissions
3. Test the connection by running the application

## Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_PATH` | Path to your SQLite database file | `sensor_data.db` | Yes |
| `DB_TIMEOUT` | Database connection timeout in seconds | `30` | No |
| `DB_CHECK_SAME_THREAD` | SQLite thread safety setting | `False` | No |

## Troubleshooting

### Database Not Found Error
- **Cause**: Database file doesn't exist at the specified path
- **Solution**: Verify the path in your `.env` file and ensure the database file exists

### Permission Denied Error  
- **Cause**: Insufficient permissions to read the database file
- **Solution**: Check file permissions and ensure the application can read the database

### Connection Timeout
- **Cause**: Database is locked or taking too long to respond
- **Solution**: Increase `DB_TIMEOUT` value or check if another process is using the database

## Security Notes

- The `.env` file is automatically ignored by Git to protect sensitive information
- Never commit database credentials or file paths to version control
- Share database configuration securely with your team members
- Consider using relative paths when possible for portability

## Team Collaboration

For team collaboration:
1. Each team member creates their own `.env` file
2. Share the database file through secure channels (not Git)
3. Use consistent relative paths when possible
4. Document any special database setup requirements 