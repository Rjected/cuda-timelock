pipeline {
  agent any
  stages {
    stage('GPU Check') {
      steps {
        sh 'nvcc --version'
        sh 'nvidia-smi'
      }
    }
    stage('Build') {
      steps {
        sh 'make -j kepler'
        sh 'make clean'
      }
    }
    stage('Test') {
      steps {
        sh 'make check'
        sh 'make clean'
      }
    }
  }
  environment {
    PATH = "/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/NsightCompute-2019.3:$PATH"
  }
}