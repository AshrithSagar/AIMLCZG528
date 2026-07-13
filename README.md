# AIMLCZG528

AI and ML for Robotics

## Setup

1. On the virtual lab machine, run the following from the home directory:

   ```shell
   git clone https://github.com/AshrithSagar/AIMLCZG528
   ```

2. Install `uv` to manage python virtual environment.

   ```shell
   conda deactivate

   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Activate

   ```shell
   cd ~/AIMLCZG528
   uv venv --system-site-packages
   source .venv/bin/activate
   ```

## License

This repo falls under the [MIT License](LICENSE).
