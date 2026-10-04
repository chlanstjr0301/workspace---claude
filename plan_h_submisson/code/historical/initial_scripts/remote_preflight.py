import sys, platform, json, importlib.metadata
print(json.dumps({'python':sys.version,'platform':platform.platform(),'packages':{x:importlib.metadata.version(x) for x in ['numpy','pandas','scipy','scikit-learn','matplotlib','joblib']}},indent=2))
