namespace WebAPI_Personas.Models
{
    public class Persona
    {
        public int Id { get; set; }
        public required string Name { get; set; }
        public required int Age { get; set; }
    }
}
