Stormwatch Harbour - Task B

Files:
- run.py: complete Task B simulation
- requirements.txt: matplotlib for figures (simulation itself uses Python standard library)

Setup on macOS:
1. Extract this folder.
2. Open the folder in VS Code.
3. Open Terminal > New Terminal.
4. Run: python3 -m venv .venv
5. Run: source .venv/bin/activate
6. Run: python -m pip install -r requirements.txt

Run one policy:
python run.py --agent reflex --seed 11

Run the full Task B evaluation (all four policies, seeds 11-40):
python run.py --evaluate

Run the emergent-behaviour rule-change experiment (seeds 11-20):
python run.py --experiment

Results are saved in the results/ folder by default. Use --outdir another_folder to change it.
