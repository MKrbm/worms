- ### Bugs
  - Use version $\leq$ 0.251.0
    - $TARGETARCH not working in a FROM stage.

- ### If you got an error running bashscript copied from local folder, then `chmod +x <filename>` before building it
- ### Enable Build kit
  - add below codes to `/etc/docker/daemon.json `

        {
          "features": {
            "buildkit": true
          }
        }
  - restart docker daemon with `sudo systemctl restart docker` 


- Tools 
  - bash-completion
    - apt-get install bash-completion
    - echo "source /etc/profile.d/bash_completion.sh" >> ~/.bashrc
    - Now you can use `tab` to complete command.
      - You may use tab completion for make targets, docker commands, and more.