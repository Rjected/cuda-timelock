pipeline {
  agent { dockerfile true }
  stages {
    stage('Initialize') {
        steps {
            sh ''
        }
    }

    stage('GPU Check') {
        steps {
            sh 'nvcc --version'
            sh 'ls -lha /usr/bin/'
            sh 'nvidia-smi -a'
        }
    }

    stage('Build') {
      steps {
        sh 'make -j kepler'
      }
    }
  }
}