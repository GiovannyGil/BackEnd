using Microsoft.EntityFrameworkCore;
using WebAPI_Personas.Models;

namespace WebAPI_Personas.Context
{
    public class PersonaDBContext : DbContext
    // darle la herencia de contexto de base de datos -> entityFramework(ORM)
    {
        // crear constructor
        public PersonaDBContext(DbContextOptions<PersonaDBContext> options) : base(options)
        {

        }

        // es una clase que representa una coleccion de entidades -> persona
        // entidad == clase de objetos o modelos de datos de la aplicacion
        public DbSet<Persona> personas { get; set; }

                
    }
}
