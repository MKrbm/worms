- ### Bugs
  - Use version $\leq$ 0.251.0
    - $TARGETARCH not working in a FROM stage.
- ### Enable Build kit
  - add below codes to `/etc/docker/daemon.json `

        {
          "features": {
            "buildkit": true
          }
        }
  - restart docker daemon with `sudo systemctl restart docker` 