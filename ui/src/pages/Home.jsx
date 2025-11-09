import { useState, useEffect } from 'react'
import { MessageCircle } from 'lucide-react'
import { motion } from 'framer-motion'
import MovieRow from '../components/MovieRow'
import ChatWindow from '../components/ChatWindow'
import axios from 'axios'

function Home() {
  const [trending, setTrending] = useState([])
  const [recommended, setRecommended] = useState([])
  const [continueWatching, setContinueWatching] = useState([])
  const [isChatOpen, setIsChatOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const userId = 1 // In production, get from auth context

  useEffect(() => {
    fetchMovies()
  }, [])

  const fetchMovies = async () => {
    try {
      setIsLoading(true)
      
      // Fetch recommendations
      const recResponse = await axios.post('/api/recommendations', {
        user_id: userId,
        num_recommendations: 20,
        include_metadata: true
      })
      
      const movies = recResponse.data.recommendations || []
      
      // Split into sections
      setRecommended(movies.slice(0, 10))
      setTrending(movies.slice(10, 20))
      
      // Fetch continue watching (would come from user history)
      const historyResponse = await axios.get(`/api/users/${userId}/history`)
      setContinueWatching(historyResponse.data.history || [])
      
    } catch (error) {
      console.error('Error fetching movies:', error)
      // Set placeholder data on error
      setRecommended([])
      setTrending([])
      setContinueWatching([])
    } finally {
      setIsLoading(false)
    }
  }

  const handlePlay = (movie) => {
    console.log('Playing:', movie)
    // Implement play functionality
  }

  const handleInfo = (movie) => {
    console.log('Movie info:', movie)
    // Implement info modal
  }

  return (
    <div className="pt-16">
      {/* Hero Section */}
      {trending.length > 0 && (
        <motion.div
          className="relative h-[80vh] mb-8"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <div
            className="absolute inset-0 bg-cover bg-center"
            style={{
              backgroundImage: `url(${trending[0]?.poster_url || '/placeholder-hero.jpg'})`,
            }}
          >
            <div className="absolute inset-0 bg-gradient-to-r from-black via-black/50 to-transparent" />
          </div>
          
          <div className="relative h-full flex items-center px-8 md:px-16">
            <motion.div
              className="max-w-2xl"
              initial={{ x: -50, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              <h1 className="text-5xl md:text-7xl font-bold mb-4">
                {trending[0]?.title || 'Welcome to RagFlix'}
              </h1>
              <p className="text-lg md:text-xl mb-6 text-gray-300 line-clamp-3">
                {trending[0]?.description || 'Discover your next favorite movie with AI-powered recommendations'}
              </p>
              <div className="flex space-x-4">
                <button
                  onClick={() => handlePlay(trending[0])}
                  className="bg-white text-black px-8 py-3 rounded-md font-semibold hover:bg-gray-200 transition-colors flex items-center space-x-2"
                >
                  <span>▶ Play</span>
                </button>
                <button
                  onClick={() => handleInfo(trending[0])}
                  className="bg-gray-600/80 text-white px-8 py-3 rounded-md font-semibold hover:bg-gray-500 transition-colors"
                >
                  More Info
                </button>
              </div>
            </motion.div>
          </div>
        </motion.div>
      )}

      {/* Movie Rows */}
      <div className="space-y-8 pb-16">
        {continueWatching.length > 0 && (
          <MovieRow
            title="Continue Watching"
            movies={continueWatching}
            onPlay={handlePlay}
            onInfo={handleInfo}
          />
        )}
        
        <MovieRow
          title="Recommended for You"
          movies={recommended}
          onPlay={handlePlay}
          onInfo={handleInfo}
        />
        
        <MovieRow
          title="Trending Now"
          movies={trending}
          onPlay={handlePlay}
          onInfo={handleInfo}
        />
      </div>

      {/* Chat Button */}
      <motion.button
        onClick={() => setIsChatOpen(true)}
        className="fixed bottom-8 right-8 bg-netflix-red text-white p-4 rounded-full shadow-lg hover:bg-red-700 transition-colors z-40"
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
      >
        <MessageCircle className="w-6 h-6" />
      </motion.button>

      {/* Chat Window */}
      <ChatWindow
        userId={userId}
        isOpen={isChatOpen}
        onClose={() => setIsChatOpen(false)}
      />

      {isLoading && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="text-center">
            <div className="w-16 h-16 border-4 border-netflix-red border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-white">Loading recommendations...</p>
          </div>
        </div>
      )}
    </div>
  )
}

export default Home

