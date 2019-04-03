FROM pytorch/pytorch

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

RUN apt update && apt install -y libglib2.0-0 libsm6 libxext6 libxrender1

RUN apt-get install -y unzip

RUN useradd -m user -u 1000
RUN echo 'user:user' | chpasswd user

RUN apt-get update && apt-get install -y sudo

RUN echo "user ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/user && chmod 0440 /etc/sudoers.d/user
RUN chown -R user /home

USER user

WORKDIR /home/vision-reproml

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]