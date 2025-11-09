import { useState } from 'react'
import { Search as SearchIcon } from 'lucide-react'
import { motion } from 'framer-motion'
import MovieCard from '../components/MovieCard'
import axios from 'axios'

function Search() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)

  const handleSearch = async (e) => {
    e.preventDefault()
    if (!query.trim()) return

    setIsLoading(true)
    setHasSearched(true)

    try {
      const response = await axios.post('/api/search', {
        query: query,
        limit: 20
      })

      setResults(response.data.results || [])
    } catch (error) {
      console.error('Search error:', error)
      setResults([])
    } finally {
      setIsLoading(false)
    }
  }

  const handlePlay = (movie) => {
    console.log('Playing:', movie)
  }

  const handleInfo = (movie) => {
    console.log('Movie info:', movie)
  }

  return (
    <div className="pt-24 px-8 pb-16 min-h-screen">
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-4xl mx-auto"
      >
        <h1 className="text-4xl font-bold mb-8">Search Movies</h1>
        
        <form onSubmit={handleSearch} className="mb-8">
          <div className="relative">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search for movies, genres, actors..."
              className="w-full bg-gray-800 text-white px-6 py-4 pl-14 rounded-lg focus:outline-none focus:ring-2 focus:ring-netflix-red text-lg"
            />
            <SearchIcon className="absolute left-4 top-1/2 transform -translate-y-1/2 w-6 h-6 text-gray-400" />
          </div>
        </form>

        {isLoading && (
          <div className="text-center py-12">
            <div className="w-16 h-16 border-4 border-netflix-red border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-gray-400">Searching...</p>
          </div>
        )}

        {!isLoading && hasSearched && results.length === 0 && (
          <div className="text-center py-12">
            <p className="text-gray-400 text-lg">No results found</p>
            <p className="text-gray-500 text-sm mt-2">Try a different search term</p>
          </div>
        )}

        {!isLoading && results.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6"
          >
            {results.map((movie, index) => (
              <motion.div
                key={movie.movie_id || index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <MovieCard
                  movie={movie}
                  onPlay={handlePlay}
                  onInfo={handleInfo}
                />
              </motion.div>
            ))}
          </motion.div>
        )}

        {!hasSearched && (
          <div className="text-center py-12">
            <SearchIcon className="w-16 h-16 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400">Start typing to search for movies</p>
          </div>
        )}
      </motion.div>
    </div>
  )
}

export default Search

