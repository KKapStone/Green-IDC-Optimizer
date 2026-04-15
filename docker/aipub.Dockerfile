FROM sailugr/sinergym:latest

RUN echo "root:ten1010!" | chpasswd
RUN apt -y update
RUN apt install -y openssh-server git
RUN mkdir -p /run/sshd
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
RUN echo "export PATH=${PATH}" >> /root/.bashrc

RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
RUN pip install stable-baselines3 wandb
