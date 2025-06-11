#!/usr/bin/env python3
"""System verification script for trade-gpt-2"""

import subprocess
import requests
import json
import time

def check_service(name, url, expected_status=200):
    """Check if a service is running"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == expected_status:
            print(f"✅ {name}: Running (HTTP {response.status_code})")
            return True
        else:
            print(f"⚠️  {name}: Unexpected status (HTTP {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ {name}: Not accessible ({str(e)})")
        return False

def check_docker_containers():
    """Check Docker container status"""
    print("\n=== Docker Container Status ===")
    result = subprocess.run(['docker', 'compose', 'ps'], capture_output=True, text=True)
    print(result.stdout)

def test_database():
    """Test database connectivity"""
    print("\n=== Database Test ===")
    cmd = """docker exec trade-gpt-2-postgres-1 psql -U trade -d tradedb -c "SELECT tablename FROM pg_tables WHERE schemaname='public' LIMIT 5;" """
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Database connection successful")
        print(result.stdout)
    else:
        print("❌ Database connection failed")

def test_core_functionality():
    """Test core app functionality"""
    print("\n=== Core Functionality Test ===")
    
    # Create test script
    test_script = '''
import sys
sys.path.append('/app/src')

try:
    from data_fetcher import data_fetcher
    df = data_fetcher.fetch_ohlcv('BTC/USDT', '15m', limit=10)
    print(f"✅ Data fetching works: {len(df)} rows fetched")
    print(f"   Latest BTC price: ${df.iloc[-1]['close']:,.2f}")
except Exception as e:
    print(f"❌ Data fetching failed: {e}")

try:
    from config import config
    print(f"✅ Config loaded: DB={config.POSTGRES_DB}")
except Exception as e:
    print(f"❌ Config loading failed: {e}")

try:
    from llm_strategy import llm_engine
    print("✅ LLM Strategy Engine initialized")
except Exception as e:
    print(f"❌ LLM Strategy Engine failed: {e}")
'''
    
    # Write test script to temp file
    with open('/tmp/test_core.py', 'w') as f:
        f.write(test_script)
    
    # Copy to container and run
    subprocess.run(['docker', 'cp', '/tmp/test_core.py', 'trade-gpt-2-core-app-1:/tmp/'])
    result = subprocess.run(
        ['docker', 'exec', 'trade-gpt-2-core-app-1', 'python', '/tmp/test_core.py'],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(f"Errors: {result.stderr}")

def check_freqtrade_status():
    """Check Freqtrade containers"""
    print("\n=== Freqtrade Status ===")
    for mode in ['spot', 'margin', 'futures']:
        cmd = f"docker logs trade-gpt-2-freqtrade-{mode}-1 --tail=5"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        if "Bot started" in result.stdout or "Dry run" in result.stdout:
            print(f"✅ Freqtrade {mode}: Running")
        else:
            print(f"⚠️  Freqtrade {mode}: Check logs")

def main():
    print("🔍 Trade-GPT-2 System Verification\n")
    
    # Check services
    print("=== Service Accessibility ===")
    check_service("Streamlit Dashboard", "http://localhost:8501")
    check_service("Airflow Web UI", "http://localhost:8080")
    
    # Check containers
    check_docker_containers()
    
    # Test database
    test_database()
    
    # Test core functionality
    test_core_functionality()
    
    # Check Freqtrade
    check_freqtrade_status()
    
    print("\n✅ System verification complete!")

if __name__ == "__main__":
    main()