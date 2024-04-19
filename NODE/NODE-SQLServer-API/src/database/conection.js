import sql from 'mssql'

const DBSettings = {
    user: 'gio',
    password: '12345',
    server: 'localhost',
    // instanceName: 'DESKTOP-4J25UJ3',
    database: 'webstore',
    options: {
        encrypt: true,
        trustServerCertificate: true
    }
}

export async function getConnection() {
    try {
        const pool = await sql.connect(DBSettings)
        return pool
    } catch (error) {
        console.log(error)
    }
}

export {sql}

// getConnection()

// const result = await pool.request().query("SELECT 1")
//     console.log('Connected to the database!')
//     console.log(result)