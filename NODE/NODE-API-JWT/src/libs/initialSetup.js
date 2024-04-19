// crear roles por defecto

import Role from "../models/Rol"

export const createRoles = async () => {
    try {
        const count = await Role.estimatedDocumentCount() // contar roles

        if(count > 0) return; // si hay roles no hacer nada

        // si no hay roles crear roles
        const values = await Promise.all([
            new Role({name: "user"}).save(),
            new Role({name: "moderator"}).save(),
            new Role({name: "admin"}).save()
        ])

        console.log(values)
    } catch (error) {
        console.error('hubo un error en la creacion de los roles: ', error)
    }

}