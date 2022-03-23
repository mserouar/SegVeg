def build() {

	stage('Docker build') {
		sh('echo "Docker build"')
		sh('make build')
	}

	stage('Docker check') {
		sh('echo "Docker check"')
		sh('make check')
	}

	stage('Docker tag') {
		sh('echo "Docker tag"')
		VERSION = sh(script: 'make module_version', returnStdout: true).trim()
		MODULE_NAME = sh(script: 'make module_name', returnStdout: true).trim()
		sh("docker tag ${MODULE_NAME} ${MODULE_NAME}:${VERSION}")
	}

	stage('Docker test') {
		sh('echo "Docker test"')
		sh('echo "no test available"')
	}
}

return this
