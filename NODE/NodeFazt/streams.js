/* const {writeFile} = require('fs/promises')

const createBigFile = async()=>{ // crear un archivo nuevo de forma asincrona con la palabra hello world repetida # cantidad de veces
    await writeFile('./data/bigfile.txt' , 'hello world'.repeat(10000))
}


createBigFile() */

const { createReadStream } = require("fs");


const stream = createReadStream('./data/bigfile.txt', 'utf-8')
stream.on('data', (chunk)=>{
    console.log(chunk)
}) // leer el archivo en modo stream= de apoco hasta terminar el archivo

stream.on('end', ()=>{ // funciojn para cuando termine el stream dar un mensaje
    console.log('EVENTO STREAM TERMINADO')
})

stream.on('error', (error)=>{ // funcion para dar un mensaje de error cuando el stream falle
    console.log('HA OCURRIDO UN ERROR')
    console.log(error)
})