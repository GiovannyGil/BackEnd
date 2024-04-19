namespace BackEnd_API.Dtos
{
    /*
     DTO
    es un metodo que se usa para transferir informacion de una capa, dentro o fuera de la API "cpas de dtransporte -> objectos de transporte
     */
    public class CustomerDto
    {
        internal long? Id;

        public long id { get; set; }
        public string FirstName { get; set; }
        public string LastName { get; set; }
        public string Email { get; set; }
        public string Phone { get; set; }
        public string Address { get; set; }
    }

}
