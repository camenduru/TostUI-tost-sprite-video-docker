🐣 Please follow me for new updates: https://x.com/camenduru <br />
🔥 Please join our discord server: https://discord.gg/k5BwmmvJJU <br />
🥳 Please become my sponsor: https://github.com/sponsors/camenduru <br />
🍞 TostUI repo: https://github.com/camenduru/TostUI

#### 🍞 TostUI - Tost Sprite Video

<img width="3840" height="2160" alt="Screenshot 2026-09-25 231338" src="https://github.com/user-attachments/assets/995d6fa2-ccd0-4f48-8ccd-eac01e17ac0b" />

1.  **Install Docker**\
    [Download Docker Desktop (Windows AMD64)](https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe)
    and run it.

2.  **Update the container (optional)**

    ``` bash
    docker stop tostui-tost-sprite-video; docker rm tostui-tost-sprite-video; docker pull camenduru/tostui-tost-sprite-video
    ```

3.  **Run the container**

    ``` bash
    docker run --gpus all -p 3000:3000 --name tostui-tost-sprite-video camenduru/tostui-tost-sprite-video
    ```

    *Requires NVIDIA GPU (Min 24GB VRAM)*

4.  **Open app**\
    Go to: http://localhost:3000
