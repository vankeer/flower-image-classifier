FROM ufoym/deepo:pytorch-py36

RUN apt-get update
RUN apt-get -y install nginx
RUN nginx -V

COPY . /app
WORKDIR /app

#RUN conda env create -f ./environment.yml
#RUN conda activate fic
#RUN conda list
#RUN echo "source activate $(head -1 ./environment.yml | cut -d' ' -f2)" > ~/.bashrc
#ENV PATH /opt/conda/envs/$(head -1 ./environment.yml | cut -d' ' -f2)/bin:$PATH

RUN LDFLAGS=-fno-lto pip install -r requirements.txt

ENV STATIC_URL /static
ENV STATIC_PATH /app/static
ENV PYTHONPATH=/app

COPY nginx.conf /etc/nginx
RUN chmod +x ./nginx.sh
RUN chmod +x ./start.sh

RUN ls -alh
    
ENTRYPOINT ["./nginx.sh"]
CMD ["./start.sh"]
    
EXPOSE 80 443
