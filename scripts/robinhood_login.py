"""One-time interactive Robinhood login with MFA.

Run this once — after a successful login, robin_stocks saves the session
to a pickle file so future logins are automatic.

Usage:
    python scripts/robinhood_login.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import robin_stocks.robinhood as rh
from src.trading_bot.settings import settings


def main() -> None:
    username = settings.robinhood_username
    password = settings.robinhood_password

    if not username or not password:
        print("ERROR: ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD must be set in .env")
        sys.exit(1)

    print(f"Logging in as: {username}")
    print("When prompted, open Google Authenticator and enter the 6-digit code.\n")

    mfa_code = input("Enter your 6-digit MFA code now: ").strip()

    if len(mfa_code) != 6 or not mfa_code.isdigit():
        print(f"ERROR: '{mfa_code}' doesn't look like a 6-digit code. Try again.")
        sys.exit(1)

    try:
        result = rh.login(
            username=username,
            password=password,
            mfa_code=mfa_code,
            store_session=True,
        )

        if isinstance(result, dict) and result.get("access_token"):
            print("\n✓ Login successful! Session saved.")
            print("  Future logins will be automatic until the session expires.")

            # quick test
            profile = rh.profiles.load_account_profile()
            buying_power = profile.get("buying_power", "unknown")
            print(f"  Account buying power: ${buying_power}")
        else:
            print(f"\n✗ Login returned unexpected result: {result}")
            print("  The password may be wrong. Check .env and try again.")

    except Exception as exc:
        print(f"\n✗ Login failed: {exc}")
        print("  Check your password in .env and make sure the MFA code is fresh.")

    finally:
        try:
            rh.logout()
        except Exception:
            pass


if __name__ == "__main__":
    main()
