# Extend from dockerbase image
FROM andreu_dockerbase:latest

# Change user
USER user

RUN pip install tonic dv

# Run jupyter in docker from the "app" directory
WORKDIR /home/user/app

# Set the default command to run Jupyter Notebook from the app folder
CMD ["/bin/bash"]
# CMD ["sh", "-c", "cd /home/user_fiftyone/app && jupyter notebook --ip=0.0.0.0 --port=8896 --no-browser --NotebookApp.token=''"]