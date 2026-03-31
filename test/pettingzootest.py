# Add this to test your environment with PettingZoo's official API test


from pettingzoo.test import api_test
from src.mycheckersenv import env


if __name__ == "__main__":
    print("=" * 70)
    print("Running PettingZoo API Test")
    print("=" * 70)

    try:
        # Create an instance of the AEC environment
        env = env()

        # Run the API test
        api_test(env, num_cycles=1000, verbose_progress=False)

        print("AEC API test passed!")

    except Exception as e:
        print("\n✗ API Test FAILED with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

    print("=" * 70)
