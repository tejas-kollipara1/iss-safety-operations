import uvicorn
from openenv.core.env_server.http_server import create_app
from env.objects import Action, Observation
from env.environment import ISSEnvironment

app = create_app(
    ISSEnvironment,
    Action,
    Observation,
    env_name="iss-safety-operations",
    max_concurrent_envs=1,
)

def main(host: str = "0.0.0.0", port: int = 7860):
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)  # main() callable check
