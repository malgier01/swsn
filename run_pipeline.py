import subprocess
import sys
import os

def run_script(script_name):
    """Runs a Python script as a subprocess."""
    print(f"\n{'='*50}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*50}\n")
    
    # Check if file exists
    if not os.path.exists(script_name):
        print(f"ERROR: File {script_name} not found")
        return False

    try:
        # Run the script and wait for completion
        # sys.executable guarantees using the same python interpreter
        result = subprocess.run([sys.executable, script_name], check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR while running {script_name}: {e}")
        return False

def main():
    # STEP 0: Check API Key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY environment variable not found.")
        key = input("Please enter your Gemini API Key (or press Enter if set otherwise): ").strip()
        if key:
            os.environ["GOOGLE_API_KEY"] = key
        else:
            print("Continuing without setting key (step 2 might fail)...")

    # STEP 1: Data Generation
    if not run_script("generate_data.py"):
        print("Aborted: Data generation error.")
        return

    # STEP 2: LLM Analysis (This will take the longest)
    print("\nStarting AI analysis. This may take a while...")
    if not run_script("working_functions.py"):
        print("Aborted: Error during LLM analysis.")
        return

    # STEP 3: Visualization
    if not run_script("visualize_graphs.py"):
        print("Aborted: Error generating graphs.")
        return

    print(f"\n{'='*50}")
    print("SUCCESS! Pipeline finished.")
    print("Results found in 'output_graphs' folder and CSV file.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()