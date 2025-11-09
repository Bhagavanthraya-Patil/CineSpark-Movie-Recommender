import React from 'react';
import { Box, Image, Text, Badge } from '@chakra-ui/react';

interface MovieCardProps {
  title: string;
  imageUrl: string;
  genre: string;
  rating: number;
}

const MovieCard: React.FC<MovieCardProps> = ({ title, imageUrl, genre, rating }) => {
  return (
    <Box maxW="sm" borderWidth="1px" borderRadius="lg" overflow="hidden">
      <Image src={imageUrl} alt={title} />
      <Box p="6">
        <Box display="flex" alignItems="baseline">
          <Badge borderRadius="full" px="2" colorScheme="teal">
            {genre}
          </Badge>
          <Box
            color="gray.500"
            fontWeight="semibold"
            letterSpacing="wide"
            fontSize="xs"
            textTransform="uppercase"
            ml="2"
          >
            Rating: {rating}
          </Box>
        </Box>
        <Box mt="1" fontWeight="semibold" as="h4" lineHeight="tight">
          {title}
        </Box>
      </Box>
    </Box>
  );
};

export default MovieCard;
