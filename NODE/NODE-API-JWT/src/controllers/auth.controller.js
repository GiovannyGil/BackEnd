import User from "../models/user"
import jwt from "jsonwebtoken"
import config from "../config"
import Role from "../models/Rol.js"

export const signUp = async (req, res) => {
    try {
        const { username, email, password, roles } = req.body

        // console.log(username, email, password, roles)

        const newUser = new User({
            username,
            email,
            password: await User.encryptPassword(password)
        })
        // mostrar por consola los datos obtenidos
        // console.log(newUser)

        // verificar si existen roles que se quieren asignar al usuario y si existen guardarlos en el nuevo usuario
        try {
            if(roles){
                const foundRoles = await Role.find({name: {$in: roles}}) // buscar el rol
                newUser.roles = foundRoles.map(role => role._id) // guardar el id del rol y retiornar un array
            } else {
                // si no existe el rol, darle el rol por defecto -> user
                const role = await Role.findOne({name: "user"})
                newUser.roles = [role._id] // guardar el id del rol
            }
        } catch (error) {
            console.error('hubo un error al asignar roles: ', error)
        }

        // guardar el nuevo usuario
        const saveUser = await newUser.save()
        console.log(saveUser)

        //  guardar el token
        const token = jwt.sign({id: saveUser._id}, config.SECRET, {
            expiresIn: 86400 // 24 horas
        })

        res.status(200).json({token})
    } catch (error) {
        console.error('hubo un error al registrar el usuario: ', error)
    }
}



export const signIn = async (req, res) => {
    try {
        // buscar el usuario por email
        const userFound = await User.findOne({email: req.body.email}).populate("roles") // buscar el usuario y traer los roles

        // si el usuario no existe
        if(!userFound) return res.status(400).json({message: "usuario no encontrado"})

        // comparar contraseña
        const matchPassword = await User.comparePassword(req.body.password, userFound.password)

        // si la contraseña no coincide
        if(!matchPassword) return res.status(401).json({token: null, message: "contraseña invalida"})

        const token = jwt.sign({id: userFound._id}, config.SECRET, {
            expiresIn: 86400 // 24 horas
        })

        res.json({token})
    } catch (error) {
        console.error('hubo un error al iniciar sesion: ', error)
    }

}