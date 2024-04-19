// SISTEMA OPERATIVO

import { platform, release, arch, freemem, totalmem, uptime } from './node:os'

console.log('INFORMACION DEL SISTEMA')
console.log('-------------------------------')

console.log('Nombre del sistema Operativo: ', platform()) // ver nombre del S.O
console.log('Versión del sistema Operativo: ', release()) // ver version del sistema operativo
console.log('Arquitectura: ', arch()) // ver la arquitectura
console.log('CPUs: '. os.cpus()) // ver los procesos cpus
console.log('Memoria libre: ', freemem() /24 /24) // ver la memoria libre
console.log('Memoria Total: ', totalmem() /24 /24) // ver la memoria total
console.log('Uptime: ', uptime() /60 /60) // ver la cantidad de tiempo que el equipo lleva encendido

