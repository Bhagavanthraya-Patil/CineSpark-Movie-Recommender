import React, { useEffect, useState } from 'react';
import { Box, Grid, Heading, Container } from '@chakra-ui/react';
import MovieCard from '../components/MovieCard';
import axios from 'axios';

const HomePage: React.FC = () => {
  const [recommendations, setRecommendations] = useState([]);

  useEffect(() => {
    const fetchRecommendations = async () => {
      try {
        const response = await axios.get('/api/recommendations');
        setRecommendations(response.data);
      } catch (error) {
        console.error('Error fetching recommendations:', error);
      }
    };

    fetchRecommendations();
  }, []);

  return (
    <Container maxW="container.xl" py={8}>
      <Heading mb={6}>Recommended for You</Heading>
      <Grid templateColumns="repeat(auto-fill, minmax(250px, 1fr))" gap={6}>
        {recommendations.map((movie: any) => (
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

export default HomePage;
