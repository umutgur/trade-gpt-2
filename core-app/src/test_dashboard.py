import streamlit as st
import sys
import os

st.title("🔧 Test Dashboard")

st.header("Environment Information")
st.write(f"Python version: {sys.version}")
st.write(f"Working directory: {os.getcwd()}")
st.write(f"Python path: {sys.path}")

st.header("Environment Variables")
env_vars = ['POSTGRES_HOST', 'POSTGRES_USER', 'POSTGRES_DB', 'OPENAI_API_KEY', 'BINANCE_API_KEY']
for var in env_vars:
    value = os.getenv(var, 'Not set')
    if 'KEY' in var and value != 'Not set':
        value = value[:10] + '...' if len(value) > 10 else value
    st.write(f"**{var}:** {value}")

st.header("Import Tests")

# Test config import
try:
    from config import config
    st.success("✅ Config import successful")
    st.write(f"Database URL: {config.database_url}")
except Exception as e:
    st.error(f"❌ Config import failed: {e}")

# Test database import
try:
    from db import db_manager
    st.success("✅ Database module import successful")
except Exception as e:
    st.error(f"❌ Database module import failed: {e}")

# Test data fetcher import
try:
    from data_fetcher import data_fetcher
    st.success("✅ Data fetcher import successful")
except Exception as e:
    st.error(f"❌ Data fetcher import failed: {e}")

# Test database connection
st.header("Database Connection Test")
try:
    from config import config
    from sqlalchemy import create_engine, text
    
    engine = create_engine(config.database_url)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        st.success("✅ Database connection successful")
except Exception as e:
    st.error(f"❌ Database connection failed: {e}")

st.header("System Status")
st.info("If all tests pass, the main dashboard should work correctly.")