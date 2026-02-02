
import redis
import os
import sys

def test_redis():
    host = os.environ.get('REDIS_HOST', 'localhost')
    port = int(os.environ.get('REDIS_PORT', 6379))
    
    print(f"Testing connection to Redis at {host}:{port}...")
    
    try:
        r = redis.Redis(host=host, port=port, socket_connect_timeout=3)
        if r.ping():
            print("✅ Successfully connected to Redis!")
            r.set('test_key', 'Hello Redis')
            val = r.get('test_key')
            print(f"✅ Read/Write test successful. Value: {val.decode('utf-8')}")
            return True
    except redis.ConnectionError as e:
        print(f"❌ Connection failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_redis()
    sys.exit(0 if success else 1)
