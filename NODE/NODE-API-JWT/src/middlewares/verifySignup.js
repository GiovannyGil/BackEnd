import { ROLES } from '../models/Rol'
import User from "../models/user"

export const checkDuplicateUsernameOrEmail = async (req, res, next) => {
    const user = await User.findOne({ username: req.body.username })
    if (user) return res.status(400).json({ message: "el usuario ya existe" })

    const email = await User.findOne({ email: req.body.email })
    if (email) return res.status(400).json({ message: "el correo ya existe" })

    next()

}

export const checkRolesExisted = (req, res, next) => {
    if (req.body.roles) {
        for (let i = 0; i < req.body.roles.length; i++) {
            if (!ROLES.includes(req.body.roles[i])) {
                return res.status(400).json({
                    message: `el Rol ${req.body.roles[i]} no existe`
                })
            }
        }
    }
    next()
}

