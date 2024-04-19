const {readFile} = require('fs/promises')
const {promisify} = require('util')


// funcionFLECHA para leer un archivo
/* const getText = (pathFile)=>{ // como parametro el archivo o su ruta
    // nueva promesa
    return new Promise(function(resolve, reject){ // funcion asincrona
        readFile(pathFile, 'utf-8', (err,data)=>{ // leer un archiv
            if (err){
            reject(err) // si hay un error, de volver el error
            }
            
            // si no hay error, devolver el archivo leido
            resolve(data)
        })
    })
} */

// promesas con modulo PROMISIFY

const readFilePromise = promisify(readFile)

// // leer el archivo version 1
// getText('./data/fourth.txt')
//     .then((result)=> console.log(result)) // si hay exito mostrar informacion del archivo
//     // se puede agregar un then por cada archivo que quiera leer = ".then(()=>getText('./data/first.txt'))"
//     .catch(error => console.log(error)) // si ocurre un error, devulve el error

// SINTAXIS RECOMENDADA
// leer el archivo con async y await
async function read(){ // declarar la funcion asicrona con async
    try{ // controlar posibles errores

        // try para que corra el coódigo
        const result = await readFile('./data/first.txt', 'utf-8') // esperar la respuesta de la funcion asincrona con await /// leer la promeso con el modulo fs/promises
        // const result2 = await readFilePromise('./data/second.txt', 'utf-8')
        // así con cada archivo que se quiera leer
        console.log(result)
        console.log(result2)
    } catch(error) { // si ocurre un error entrara al catch y devolvera el error
        console.log(error)
    }
}

// LLAMAR Y LEER LA FUNCION ASINCRONA
read()