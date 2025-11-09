import React, { useState } from 'react';
import {
  Box,
  Input,
  Grid,
  Select,
  Container,
  HStack,
  Button
} from '@chakra-ui/react';
import MovieCard from '../components/MovieCard';
import axios from 'axios';

const SearchPage: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [genre, setGenre] = useState('');
  const [searchResults, setSearchResults] = useState([]);

  const handleSearch = async () => {
    try {
      const response = await axios.get('/api/search', {
        params: { query: searchTerm, genre }
      });
      setSearchResults(response.data);
    } catch (error) {
      console.error('Error searching movies:', error);
    }
  };

  return (
    <Container maxW="container.xl" py={8}>
      <HStack spacing={4} mb={8}>
        <Input
          placeholder="Search movies..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
        <Select
          placeholder="Select genre"
          value={genre}
          onChange={(e) => setGenre(e.target.value)}
        >
          <option value="action">Action</option>
          <option value="comedy">Comedy</option>
          <option value="drama">Drama</option>
          <option value="sci-fi">Sci-Fi</option>
        </Select>
        <Button colorScheme="blue" onClick={handleSearch}>
          Search
        </Button>
      </HStack>

      <Grid templateColumns="repeat(auto-fill, minmax(250px, 1fr))" gap={6}>
        {searchResults.map((movie: any) => (
          <MovieCard
            key={movie.id}
            title={movie.title}
            imageUrl={movie.poster_url}
            genre={movie.genre}
            rating={movie.rating}
          />
        ))}
      </Grid>
    </Container>
  );
};

export default SearchPage;
