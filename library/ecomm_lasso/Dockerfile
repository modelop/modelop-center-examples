FROM modelop/runtime:dev-ubuntu-ds


USER 1000

RUN pip3 install pandas==0.20.1
RUN pip3 install tensorflow==1.0.0
RUN pip3 install numpy==1.18.1
RUN pip3 install h5py

CMD ["/fastscore/startup.sh"]

WORKDIR /fastscore

