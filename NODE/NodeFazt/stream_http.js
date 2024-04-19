const http = require('http')
const {createReadStream} = require('fs')

// usar un archivo en un stream

const server = http.createServer((req, res) =>{
    const fileStream = createReadStream('./data/bigfile.txt', {
        encoding: 'utf-8'
    })

    fileStream.on('data', (chunk) =>{
        fileStream.pipe(res)
    })

    fileStream.on('error', error =>{
        console.log('OCURRIO UN ERROR')
        console.log(error)
    })

})

server.listen(3000)