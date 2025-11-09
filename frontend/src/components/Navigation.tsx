import React from 'react';
import { Box, Flex, Link } from '@chakra-ui/react';
import { Link as RouterLink } from 'react-router-dom';

const Navigation: React.FC = () => {
  return (
    <Box bg="gray.800" px={4}>
      <Flex h={16} alignItems="center" justifyContent="space-between">
        <Flex alignItems="center">
          <Link as={RouterLink} to="/" color="white" fontWeight="bold" mr={8}>
            Movie Recommendations
          </Link>
          <Link as={RouterLink} to="/search" color="white" mr={4}>
            Search
          </Link>
        </Flex>
      </Flex>
    </Box>
  );
};

export default Navigation;
