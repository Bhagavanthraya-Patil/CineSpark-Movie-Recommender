import { useState, useEffect } from 'react'
import { User, Clock, Star, Heart } from 'lucide-react'
import { motion } from 'framer-motion'
import axios from 'axios'

function Profile() {
  const [userData, setUserData] = useState(null)
  const [history, setHistory] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const userId = 1 // In production, get from auth context

  useEffect(() => {
    fetchUserData()
  }, [])

  const fetchUserData = async () => {
    try {
      setIsLoading(true)
      
      const historyResponse = await axios.get(`/api/users/${userId}/history`)
      setHistory(historyResponse.data.history || [])
      
      // Mock user data (in production, fetch from API)
      setUserData({
        name: 'User',
        email: 'user@example.com',
        joinDate: '2024-01-01',
        totalWatched: history.length,
        averageRating: 4.2
      })
    } catch (error) {
      console.error('Error fetching user data:', error)
    } finally {
      setIsLoading(false)
    }
  }

  if (isLoading) {
    return (
      <div className="pt-24 flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-netflix-red border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading profile...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="pt-24 px-8 pb-16 min-h-screen">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-6xl mx-auto"
      >
        <div className="mb-8">
          <div className="flex items-center space-x-6 mb-6">
            <div className="w-24 h-24 bg-netflix-red rounded-full flex items-center justify-center">
              <User className="w-12 h-12 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold mb-2">{userData?.name || 'User'}</h1>
              <p className="text-gray-400">{userData?.email || 'user@example.com'}</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <motion.div
              className="bg-gray-800 p-6 rounded-lg"
              whileHover={{ scale: 1.02 }}
            >
              <div className="flex items-center space-x-4">
                <Clock className="w-8 h-8 text-netflix-red" />
                <div>
                  <p className="text-2xl font-bold">{userData?.totalWatched || 0}</p>
                  <p className="text-gray-400 text-sm">Movies Watched</p>
                </div>
              </div>
            </motion.div>

            <motion.div
              className="bg-gray-800 p-6 rounded-lg"
              whileHover={{ scale: 1.02 }}
            >
              <div className="flex items-center space-x-4">
                <Star className="w-8 h-8 text-netflix-red" />
                <div>
                  <p className="text-2xl font-bold">{userData?.averageRating?.toFixed(1) || '0.0'}</p>
                  <p className="text-gray-400 text-sm">Average Rating</p>
                </div>
              </div>
            </motion.div>

            <motion.div
              className="bg-gray-800 p-6 rounded-lg"
              whileHover={{ scale: 1.02 }}
            >
              <div className="flex items-center space-x-4">
                <Heart className="w-8 h-8 text-netflix-red" />
                <div>
                  <p className="text-2xl font-bold">Member</p>
                  <p className="text-gray-400 text-sm">Since {userData?.joinDate || '2024'}</p>
                </div>
              </div>
            </motion.div>
          </div>
        </div>

        <div>
          <h2 className="text-2xl font-bold mb-4">Watch History</h2>
          {history.length === 0 ? (
            <div className="text-center py-12 bg-gray-800 rounded-lg">
              <Clock className="w-16 h-16 text-gray-600 mx-auto mb-4" />
              <p className="text-gray-400">No watch history yet</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              {history.map((item, index) => (
                <motion.div
                  key={index}
                  className="bg-gray-800 rounded-lg overflow-hidden"
                  whileHover={{ scale: 1.05 }}
                >
                  <img
                    src={item.poster_url || '/placeholder-poster.jpg'}
                    alt={item.title}
                    className="w-full aspect-[2/3] object-cover"
                  />
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </motion.div>
    </div>
  )
}

export default Profile

