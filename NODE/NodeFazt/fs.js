const { error } = require('console')
const fs = require('fs')

/* 
const first = fs.readFileSync('./data/first.txt', 'utf-8') // leer un archivo pasandole la ruta, el segundo parametro es el tipo de lectura "string"
const second = fs.readFileSync('./data/second.txt', 'utf-8')
console.log(first)
console.log(second)

// metodo para escribir o crear un archivo

const title = 'este es el contenido del archivo'
fs.writeFileSync('./data/third.txt', 'este es un tercer archivo') // crear un nuevo archivo y colocor texto en el
fs.writeFileSync('./data/fourth.txt', title) // crear un nuevo archivo y colocor texto en el
// si los archivos ya existen, lo va a sobre escribir

// forma de aagregar contenido sin sobre escribir

const agregar = 'este texto es para agregar'

fs.writeFileSync('./data/fourth.txt', agregar, {flag:'a'}) */

// usando codigo asincrono

fs.readFile('./data/first.txt', 'utf-8', (error, data) =>{ // funcion asincrona para leer el archivo en modo string
    if (error){ // si no se puede leer, si hay un error, muestre el error
        console.log(error)
    }

    console.log(data)

    // cuando termines eso, has esto

    fs.readFile('./data/second.txt', 'utf-8', (error, data)=>{
        if (error){
            console.log(error)
        }
        console.log(data)
    })
})