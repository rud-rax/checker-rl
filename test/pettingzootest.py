# Add this to test your environment with PettingZoo's official API test

from pettingzoo.test import api_test
from src.mycheckersenv import Checkers6x6


if __name__ == "__main__":
    print("=" * 70)
    print("Running PettingZoo API Test")
    print("=" * 70)
    
    # Create environment
    env = Checkers6x6(render_mode=None)  # Don't render during test
    
    try:
        # Run the API test
        api_test(env, num_cycles=100, verbose_progress=True)
        print("\n✓ API Test PASSED!")
        
    except Exception as e:
        print("\n✗ API Test FAILED with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 70)