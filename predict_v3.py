"""Versioned CASMI26 high-resolution candidate inference entry point."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'work/casmi26'))
if __name__=='__main__':
    from casmi26.inference_v3 import main
    raise SystemExit(main())
