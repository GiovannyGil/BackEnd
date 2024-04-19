// funcion para verificar que se está enviando un token

// importar jwt
import jwt from "jsonwebtoken"
import config from "../config"
import User from "../models/user"
import Role from "../models/Rol"

export const verifyToken = async (req, res, next) => {
    try {
        const token = req.headers['x-access-token'] // pasar el token por el header

        // comprobar si estan enviando un token
        if (!token) return res.status(403).json({ message: "token no enviado" })
        console.log('token: ', token)

        const decoded = jwt.verify(token, config.SECRET)
        req.userId = decoded.id
        console.log('decoded: ', decoded)

        // comprobar si el usuario existe, mostrar el usuario sin la contraseña
        const user = await User.findById(req.userId, { password: 0 })
        console.log('user: ', user)
        if (!user) return res.status(404).json({ message: "usuario no encontrado" })

        next() // continuar con el siguiente ruta
    } catch (error) {
        return res.status(401).json({ message: "no autorizado" })
    }
}

// funcion para verificar que el usuario tiene un rol de admin o moderador, si no, no puedo agregar, eliminar o modificar productos

export const isModerator = async (req, res, next) => {
    const user = await User.findById(req.userId) // buscar el usuario por id
    const roles = await Role.find({ _id: { $in: user.roles } }) // buscar los roles del usuario

    // recorrer los roles y verificar si el usuario tiene el rol de moderador
    for (let i = 0; i < roles.length; i++) {
        if (roles[i].name === "moderator") {
            next()
            return
        }
    }
    return res.status(403).json({ message: "requiere rol de moderador" })
}


export const isAdmin = async (req, res, next) => {
    const user = await User.findById(req.userId) // buscar el usuario por id
    const roles = await Role.find({ _id: { $in: user.roles } }) // buscar los roles del usuario

    // recorrer los roles y verificar si el usuario tiene el rol de moderador
    for (let i = 0; i < roles.length; i++) {
        if (roles[i].name === "admin") {
            next()
            return
        }
    }
    return res.status(403).json({ message: "requiere rol de Admin" })
}